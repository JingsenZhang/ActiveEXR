import os
import math
import torch
import torch.optim as optim

from utils import now_time, get_local_time, ids2tokens
from metrics.metrics import root_mean_square_error, mean_absolute_error, bleu_score, rouge_score


class Trainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

        self.model_name = config['model']
        self.dataset = config['dataset']
        self.method = config['method']
        self.ntokens = config['ntoken']
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.word2idx = config['word2idx']
        self.idx2word = config['idx2word']
        self.clip = config['clip']
        self.seq_max_len = config['seq_max_len']

        self.rating_criterion = config['rating_criterion']
        self.text_criterion = config['text_criterion']
        self.learner = config['learner']
        self.learning_rate = config['lr']
        self.weight_decay = config['weight_decay']
        self.optimizer = self._build_optimizer()
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.95)  # gamma: lr_decay

        self.rating_weight = config['rating_weight']
        self.text_weight = config['text_weight']
        self.l2_weight = config['l2_weight']
        self.max_rating = config['max_rating']
        self.min_rating = config['min_rating']
        self.bert_bs = config['bert_bs']

        self.endure_times = config['endure_times']
        self.checkpoint = config['checkpoint']
        self.prediction_path = config['prediction_path']

        self.rating_epochs = config['rating_epochs']
        self.rating_lr = config['rating_lr']
        self.rating_learner = config['rating_learner']
        self.rating_optimizer = self._build_optimizer_rating()

        self.seed = config['seed']
        self.unique = config['unique']

    def _build_optimizer(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.learner)
        learning_rate = kwargs.pop('learning_rate', self.learning_rate)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _build_optimizer_rating(self, **kwargs):
        r"""Init the Optimizer

        Args:
            params (torch.nn.Parameter, optional): The parameters to be optimized.
                Defaults to ``self.model.parameters()``.
            learner (str, optional): The name of used optimizer. Defaults to ``self.learner``.
            learning_rate (float, optional): Learning rate. Defaults to ``self.learning_rate``.
            weight_decay (float, optional): The L2 regularization weight. Defaults to ``self.weight_decay``.

        Returns:
            torch.optim: the optimizer
        """
        params = kwargs.pop('params', self.model.parameters())
        learner = kwargs.pop('learner', self.rating_learner)
        learning_rate = kwargs.pop('learning_rate', self.rating_lr)
        weight_decay = kwargs.pop('weight_decay', self.weight_decay)

        if learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer


class PETERTrainer(Trainer):

    def __init__(self, config, model):
        super(PETERTrainer, self).__init__(config, model)
        self.context_weight = config['context_weight']
        self.src_len = config['src_len']
        self.tgt_len = config['tgt_len']
        self.use_feature = config['use_feature']

    def predict(self, log_context_dis, topk):
        word_prob = log_context_dis.exp()  # (batch_size, ntoken)
        if topk == 1:
            context = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
        else:
            context = torch.topk(word_prob, topk, 1)[1]  # (batch_size, topk)
        return context  # (batch_size, topk)

    def pretrain_rating(self, data, model):  # train
        model.train()
        rating_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, seq, _, _, _ = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
            text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

            self.rating_optimizer.zero_grad()
            log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)
            r_loss = torch.mean(self.rating_criterion(rating_p, rating))
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            self.rating_optimizer.step()
            rating_loss += batch_size * r_loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break

        return rating_loss / total_sample

    def rating_eval(self, data, model):  # train
        model.eval()
        rating_loss = 0.
        total_sample = 0
        with torch.no_grad():
            while True:
                user, item, rating, seq, _, _, _ = data.next_batch()  # (batch_size, seq_len), data.step += 1
                batch_size = user.size(0)
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

                log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)
                r_loss = torch.mean(self.rating_criterion(rating_p, rating))
                rating_loss += batch_size * r_loss.item()
                total_sample += batch_size

                if data.step == data.total_step:
                    break

        return rating_loss / total_sample

    def train_loop_rating(self, cur_train_data, cur_val_data, model):
        best_val_loss = float('inf')
        endure_count = 0
        best_epoch = 0
        model_path = ''
        for epoch in range(1, self.rating_epochs + 1):
            print(now_time() + 'epoch {}'.format(epoch))
            train_r_loss = self.pretrain_rating(cur_train_data, model)
            print(now_time() + 'train rating loss {:4.4f} on unlabel_data'.format(train_r_loss))
            val_r_loss = self.rating_eval(cur_val_data, model)
            print(now_time() + 'valid rating loss {:4.4f} on valid_data'.format(val_r_loss))
            if val_r_loss < best_val_loss:
                best_val_loss = val_r_loss
                saved_model_file = '{}-{}-{}-{}-{}-{}.pt'.format(self.model_name, self.dataset, self.method,
                                                                 self.unique, self.seed, get_local_time())
                model_path = os.path.join(self.checkpoint, saved_model_file)
                torch.save(model.state_dict(), model_path)
                print(now_time() + 'Save the best model: ' + model_path)
                best_epoch = epoch
            else:
                endure_count += 1
                print(now_time() + 'Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                    break
        print(now_time() + 'best epoch: {}'.format(best_epoch))
        return model_path

    def train(self, data, model):  # train
        # Turn on training mode which enables dropout.
        model.train()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        total_loss = 0.
        total_sample = 0
        while True:
            user, item, rating, seq, feature, prob, _ = data.next_batch()  # (batch_size, seq_len), data.step += 1
            batch_size = user.size(0)
            user = user.to(self.device)  # (batch_size,)
            item = item.to(self.device)
            rating = rating.to(self.device)
            seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)     .t()
            feature = feature.t().to(self.device)  # (1, batch_size)
            prob = prob.to(self.device)
            if self.use_feature:
                text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
            else:
                text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            self.optimizer.zero_grad()

            # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)
            # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
            context_dis = log_context_dis.unsqueeze(0).repeat((self.tgt_len - 1, 1, 1))
            c_mid_loss = torch.mean(
                self.text_criterion(context_dis.view(-1, self.ntokens), seq[1:-1].reshape((-1,))).reshape(batch_size,
                                                                                                          -1), dim=1)
            c_loss = torch.mean(prob * c_mid_loss)
            t_mid_loss = torch.mean(
                self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[1:].reshape((-1,))).reshape(batch_size,
                                                                                                          -1), dim=1)
            t_loss = torch.mean(prob * t_mid_loss)
            r_loss = torch.mean(prob * self.rating_criterion(rating_p, rating))
            loss = self.rating_weight * r_loss + self.context_weight * c_loss + self.text_weight * t_loss
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
            self.optimizer.step()
            context_loss += batch_size * c_loss.item()
            text_loss += batch_size * t_loss.item()
            rating_loss += batch_size * r_loss.item()
            total_loss += batch_size * loss.item()
            total_sample += batch_size
            if data.step == data.total_step:
                break

        return rating_loss / total_sample, text_loss / total_sample, context_loss / total_sample, total_loss / total_sample

    def evaluate(self, data, cur_best_model):  # valid and test based on loss
        # Turn on evaluation mode which disables dropout.
        cur_best_model.eval()
        context_loss = 0.
        text_loss = 0.
        rating_loss = 0.
        total_loss = 0.
        total_sample = 0
        rating_predict = []
        with torch.no_grad():
            while True:
                user, item, rating, seq, feature, _, _ = data.next_batch()  # (batch_size, seq_len), data.step += 1
                batch_size = user.size(0)
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                rating = rating.to(self.device)
                seq = seq.t().to(self.device)  # (tgt_len + 1, batch_size)
                feature = feature.t().to(self.device)  # (1, batch_size)
                if self.use_feature:
                    text = torch.cat([feature, seq[:-1]], 0)  # (src_len + tgt_len - 2, batch_size)
                else:
                    text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)

                # (tgt_len, batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                log_word_prob, log_context_dis, rating_p, _ = cur_best_model(user, item, text)
                rating_predict.extend(rating_p.tolist())
                # (batch_size, ntoken) -> (tgt_len - 1, batch_size, ntoken)
                context_dis = log_context_dis.unsqueeze(0).repeat((self.tgt_len - 1, 1, 1))
                c_loss = torch.mean(self.text_criterion(context_dis.view(-1, self.ntokens), seq[1:-1].reshape((-1,))))
                r_loss = torch.mean(self.rating_criterion(rating_p, rating))
                t_loss = torch.mean(self.text_criterion(log_word_prob.view(-1, self.ntokens), seq[1:].reshape((-1,))))
                loss = self.rating_weight * r_loss + self.context_weight * c_loss + self.text_weight * t_loss

                context_loss += batch_size * c_loss.item()
                text_loss += batch_size * t_loss.item()
                rating_loss += batch_size * r_loss.item()
                total_loss += batch_size * loss.item()
                total_sample += batch_size

                if data.step == data.total_step:
                    break

        return rating_loss / total_sample, text_loss / total_sample, context_loss / total_sample, total_loss / total_sample

    def generate(self, data, cur_best_model, cal_bert):  # generate explanation & evaluate on metrics
        # Turn on evaluation mode which disables dropout.
        cur_best_model.eval()
        idss_predict = []
        context_predict = []
        rating_predict = []
        with torch.no_grad():
            while True:
                user, item, _, seq, feature, _, _ = data.next_batch()
                user = user.to(self.device)  # (batch_size,)
                item = item.to(self.device)
                bos = seq[:, 0].unsqueeze(0).to(self.device)  # (1, batch_size)
                feature = feature.t().to(self.device)  # (1, batch_size)
                if self.use_feature:
                    text = torch.cat([feature, bos], 0)  # (src_len - 1, batch_size)
                else:
                    text = bos  # (src_len - 1, batch_size)
                start_idx = text.size()[0]
                for idx in range(self.seq_max_len):
                    # produce a word at each step
                    if idx == 0:  # predict word from <bos>
                        # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
                        log_word_prob, log_context_dis, rating_p, _ = cur_best_model(user, item, text, False)
                        rating_predict.extend(rating_p.tolist())
                        context = self.predict(log_context_dis, topk=self.seq_max_len)  # (batch_size, seq_max_len)
                        context_predict.extend(context.tolist())
                    else:
                        log_word_prob, _, _, _ = cur_best_model(user, item, text, False, False,
                                                                False)  # (batch_size, ntoken)
                    word_prob = log_word_prob.exp()  # (batch_size, ntoken)
                    word_idx = torch.argmax(word_prob,
                                            dim=1)  # (batch_size,), pick the one with the largest probability
                    text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
                ids = text[start_idx:].t().tolist()  # (batch_size, seq_len)
                idss_predict.extend(ids)

                if data.step == data.total_step:
                    break

        # rating
        predicted_rating = [(r, p) for (r, p) in zip(data.rating.tolist(), rating_predict)]
        RMSE = root_mean_square_error(predicted_rating, self.max_rating, self.min_rating)
        print(now_time() + 'RMSE {:7.4f}'.format(RMSE))
        MAE = mean_absolute_error(predicted_rating, self.max_rating, self.min_rating)
        print(now_time() + 'MAE {:7.4f}'.format(MAE))
        # text
        tokens_test = [ids2tokens(ids[1:], self.word2idx, self.idx2word) for ids in data.seq.tolist()]
        tokens_predict = [ids2tokens(ids, self.word2idx, self.idx2word) for ids in idss_predict]
        BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
        print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
        BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
        print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))

        text_test = [' '.join(tokens) for tokens in tokens_test]
        text_predict = [' '.join(tokens) for tokens in tokens_predict]
        tokens_context = [' '.join([self.idx2word[i] for i in ids]) for ids in context_predict]
        ROUGE = rouge_score(text_test, text_predict)  # a dictionary
        for (k, v) in ROUGE.items():
            print(now_time() + '{} {:7.4f}'.format(k, v))
        text_out = ''
        for (true_r, real, ctx, pre_r, fake) in zip(data.rating.tolist(), text_test, tokens_context,
                                                    rating_predict,
                                                    text_predict):  # format: ground_truth|context|explanation
            text_out += '{}\n{}\n{}\n{}\n{}\n\n'.format(true_r, real, ctx, pre_r, fake)
        return text_out, RMSE, MAE, BLEU1, BLEU4, ROUGE

    def train_loop(self, cur_train_data, cur_val_data, model):
        best_val_loss = float('inf')
        best_epoch = 0
        endure_count = 0
        model_path = ''
        for epoch in range(1, self.epochs + 1):
            print(now_time() + 'epoch {}'.format(epoch))
            rating_loss, text_loss, context_loss, train_loss = self.train(cur_train_data, model)
            print(
                now_time() + 'rating loss {:4.4f} | text ppl {:4.4f} | context ppl {:4.4f} | total loss {:4.4f} on train'.format(
                    rating_loss, math.exp(text_loss), math.exp(context_loss), train_loss))
            rating_loss, text_loss, context_loss, val_loss = self.evaluate(cur_val_data, model)
            print(
                now_time() + 'rating loss {:4.4f} | text ppl {:4.4f} | context ppl {:4.4f} | total loss {:4.4f} on valid'.format(
                    rating_loss, math.exp(text_loss), math.exp(context_loss), val_loss))

            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saved_model_file = '{}-{}-{}-{}-{}-{}.pt'.format(self.model_name, self.dataset, self.method,
                                                                 self.unique, self.seed, get_local_time())
                model_path = os.path.join(self.checkpoint, saved_model_file)
                print(now_time() + 'Save the best model' + model_path)
                torch.save(model.state_dict(), model_path)
                best_epoch = epoch
            else:
                endure_count += 1
                print(now_time() + 'Endured {} time(s)'.format(endure_count))
                if endure_count == self.endure_times:
                    print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
                    break
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                # scheduler.step()
                # print(now_time() + 'Learning rate set to {:2.8f}'.format(scheduler.get_last_lr()[0]))
        print(now_time() + 'best epoch: {}'.format(best_epoch))
        return model_path

    def test_loop(self, cur_test_data, cur_best_model, cal_bert):
        text_o, test_RMSE, test_MAE, BLEU1, BLEU4, ROUGE = self.generate(cur_test_data, cur_best_model, cal_bert)
        save_generated_path = self.prediction_path + get_local_time() + '.txt'
        with open(save_generated_path, 'w', encoding='utf-8') as f:
            f.write(text_o)
        print(now_time() + 'Generated text saved to ({})'.format(save_generated_path))

        return test_RMSE, test_MAE, BLEU1, BLEU4, ROUGE
