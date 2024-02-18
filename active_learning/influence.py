import torch
import torch.autograd as autograd
import random

from model_utils import cal_loss_peter, gen_seq_peter, gen_seq_peter_topk


class IF(object):
    def __init__(self, model, reference_data, train_data, candidate_data, config):
        self.config = config
        self.model = model
        self.device = config['device']
        self.damp = config['damp']
        self.scale = config['scale']
        self.recursion_depth = config['depth']
        self.reference_data = reference_data
        self.train_data = train_data
        self.candidate_data = candidate_data
        self.candidate_size = candidate_data.sample_num
        self.train_idxs = [i for i in range(train_data.sample_num)]
        self.train_sample_num = config['train_num']
        self.top_K = config['topK']
        self.calculate_loss = cal_loss_peter
        self.generate_seq = gen_seq_peter
        self.generate_seq_topk = gen_seq_peter_topk
        self.use_gpu = True
        self.model.eval()
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def get_influence_function(self):
        influences = []
        s_test = self.s_test()
        for i in range(self.candidate_data.sample_num):
            data_point = self.candidate_data.get_part_data([i])
            if self.config['topK']:
                grad_z = self.cal_expected_grad_topk(data_point)
            else:
                grad_z = self.cal_expected_grad_largest(data_point)
            influence = sum([torch.sum(k * j).item()
                             for k, j in zip(grad_z, s_test)])
            influences.append(influence)
        return torch.tensor(influences).to(self.device)

    # batch calculation
    def get_influence_function_batch(self):
        influences = []
        s_test = self.s_test()
        while True:
            user, item, rating, seq, fea, _, _ = self.candidate_data.next_batch()
            data_points = {'user': user, 'item': item, 'rating': rating, 'seq': seq, 'feature': fea}
            cur_batch_size = user.size()[0]
            # print('cur_batch_size', cur_batch_size)
            if self.config['topK']:
                grad_z_batch = self.cal_expected_grad_topk_batch(data_points, cur_batch_size)
            else:
                grad_z_batch = self.cal_expected_grad_largest_batch(data_points, cur_batch_size)  # list(batch, params)
            influence_batch = [sum([torch.sum(k * j).item() for k, j in zip(grad_z, s_test)])
                               for grad_z in grad_z_batch]
            influences.extend(influence_batch)
            if self.candidate_data.step == self.candidate_data.total_step:
                break
        return torch.tensor(influences).to(self.device)

    def get_grad_dataset(self, data):
        tot_grad = None
        while True:
            user, item, rating, seq, feature, _, _ = data.next_batch()  # (batch_size, seq_len), data.step += 1
            data_points = {}
            data_points['user'] = user
            data_points['item'] = item
            data_points['rating'] = rating
            data_points['seq'] = seq
            data_points['feature'] = feature
            cur_batch_size = user.size()[0]
            weight = cur_batch_size / data.sample_num
            grad = self.grad_z(data_points)
            grad = [g * weight for g in grad]
            if tot_grad == None:
                tot_grad = grad
            else:
                tot_grad = [g1 + g2 for g1, g2 in zip(tot_grad, grad)]
            if data.step == data.total_step:
                break
        return tot_grad

    def grad_z(self, data_points):
        """
        calculate the gradient of a set of data points (or a data point)
        Args:
            data_point: the set of data points (preprocessed)  e.i. data_dict

        Returns:
            gradient list
        """
        loss, _ = self.calculate_loss(data_points, self.model, self.config)
        dL = list(autograd.grad(loss, self.params))
        return dL

    def hessian_vector_product(self, y, w, v):
        """
        calculate the hvp based on train data sampled randomly.
        :param y: model loss
        :param w: model params
        :param v: h_estimate
        :return: hvp
        """
        if len(w) != len(v):
            raise (ValueError("w and v must have the same length."))
        first_grads = autograd.grad(y, w, create_graph=True)
        # Elementwise products
        elemwise_products = 0
        for grad_elem, v_elem in zip(first_grads, v):
            elemwise_products += torch.sum(grad_elem * v_elem)
            # Second backpropagation
        return_grads = list(autograd.grad(elemwise_products, w))
        return return_grads

    def s_test(self):

        v = self.get_grad_dataset(self.reference_data)
        h_estimate = v.copy()  # S_test0
        for i in range(self.recursion_depth):
            sampled_train_idxs = random.sample(self.train_idxs, self.train_sample_num)
            sampled_train_data = self.train_data.get_part_data(sampled_train_idxs)
            loss, _ = self.calculate_loss(sampled_train_data, self.model, self.config, use_gpu=self.use_gpu)
            hv = self.hessian_vector_product(loss, self.params, h_estimate)
            with torch.no_grad():
                h_estimate = [
                    _v + (1 - self.damp) * _h_e - _hv / self.scale
                    for _v, _h_e, _hv in zip(v, h_estimate, hv)]
        h_estimate = [_h_e / self.scale for _h_e in h_estimate]
        return h_estimate

    def cal_expected_grad_largest(self, data_point):
        seq, _ = self.generate_seq(data_point, self.model, self.config)
        data_point['seq'] = torch.tensor(seq, dtype=torch.int64).contiguous()
        _, t_loss = self.calculate_loss(data_point, self.model, self.config)  # (1*16, )
        grad = list(autograd.grad(torch.mean(t_loss), self.params, allow_unused=True))
        expectation = [g if g is not None
                       else torch.tensor(0)
                       for g in grad]
        return expectation

    def cal_expected_grad_largest_batch(self, data_points, cur_batch_size):
        seq, _ = self.generate_seq(data_points, self.model, self.config)
        data_points['seq'] = torch.tensor(seq, dtype=torch.int64).contiguous()
        _, t_loss = self.calculate_loss(data_points, self.model, self.config, use_gpu=self.use_gpu)
        t_loss = t_loss.reshape(cur_batch_size, -1).mean(1)  # (batch*16, )->(batch, 16)->(batch,)
        jacobi = torch.eye(cur_batch_size).to(self.config['device'])  # (batch, batch)
        grad_list = list(autograd.grad(outputs=t_loss, inputs=self.params, grad_outputs=jacobi, is_grads_batched=True,
                                       allow_unused=True))
        expectations = []
        for i in range(cur_batch_size):
            expectation_i = [g[i] if g is not None
                             else torch.tensor(0)
                             for g in grad_list]
            expectations.append(expectation_i)
        return expectations

    def cal_expected_grad_topk(self, data_point):
        seq_list = []
        for i in range(self.top_K):
            seq = self.generate_seq_topk(data_point, self.model, self.config)
            seq_list.append(seq[0])
        user = data_point['user'].repeat(self.top_K)
        item = data_point['item'].repeat(self.top_K)
        rating = data_point['rating'].repeat(self.top_K)
        seq = torch.tensor(seq_list, dtype=torch.int64).contiguous()
        feature = data_point['feature'].repeat(self.top_K, 1)
        k_data_points = {'user': user, 'item': item, 'rating': rating, 'seq': seq, 'feature': feature}
        _, t_loss = self.calculate_loss(k_data_points, self.model, self.config)
        grad = list(autograd.grad(torch.mean(t_loss), self.params, allow_unused=True))
        expectation = [g if g is not None
                       else torch.tensor(0)
                       for g in grad]
        return expectation

    def cal_expected_grad_topk_batch(self, data_points, cur_batch_size):
        seq_list = torch.tensor([], dtype=torch.int64)
        for i in range(self.top_K):
            seq = self.generate_seq_topk(data_points, self.model, self.config)  # list [bs, seq_len]
            seq_list = torch.cat((seq_list, torch.tensor(seq, dtype=torch.int64)), dim=1)  # (batch, seq_len*k)
        user = data_points['user'].repeat_interleave(self.top_K)  # (batch,)->(batch*k,)
        item = data_points['item'].repeat_interleave(self.top_K)  # (batch,)->(batch*k,)
        rating = data_points['rating'].repeat_interleave(self.top_K)  # (batch,)->(batch*k,)
        seq = seq_list.reshape(cur_batch_size * self.top_K, -1)  # (batch*k, seq_len)
        feature = data_points['feature'].repeat_interleave(self.top_K, dim=0)  # (batch*k, 1)
        k_data_points = {'user': user, 'item': item, 'rating': rating, 'seq': seq, 'feature': feature}
        _, t_loss = self.calculate_loss(k_data_points, self.model, self.config, use_gpu=self.use_gpu)
        t_loss = t_loss.reshape(cur_batch_size, -1).mean(1)  # (batch*topk*16, )->(batch, topk*16)->(batch,)
        jacobi = torch.eye(cur_batch_size).to(self.config['device'])  # (batch, batch)
        grad_list = list(autograd.grad(outputs=t_loss, inputs=self.params, grad_outputs=jacobi, is_grads_batched=True,
                                       allow_unused=True))
        expectations = []
        for i in range(cur_batch_size):
            expectation_i = [g[i] if g is not None
                             else torch.tensor(0)
                             for g in grad_list]
            expectations.append(expectation_i)
        return expectations
