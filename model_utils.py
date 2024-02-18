import torch


def cal_loss_peter(data_points, model, config, use_gpu=True):
    user = data_points['user'].to(config['device'])
    item = data_points['item'].to(config['device'])
    rating = data_points['rating'].to(config['device'])
    seq = data_points['seq'].t().to(config['device'])
    text = seq[:-1]  # (src_len + tgt_len - 2, batch_size)
    with torch.backends.cudnn.flags(enabled=use_gpu):
        log_word_prob, log_context_dis, rating_p, _ = model(user, item, text)
    context_dis = log_context_dis.unsqueeze(0).repeat((config['tgt_len'] - 1, 1, 1))
    c_loss = config['text_criterion'](context_dis.view(-1, config['ntoken']), seq[1:-1].reshape((-1,)))
    r_loss = config['rating_criterion'](rating_p, rating)
    t_loss = config['text_criterion'](log_word_prob.view(-1, config['ntoken']), seq[1:].reshape((-1,)))
    loss = config['rating_weight'] * torch.mean(r_loss) + config['context_weight'] * torch.mean(c_loss) + config[
        'text_weight'] * torch.mean(t_loss)

    return loss, t_loss


def gen_seq_peter(data_points, model, config):
    seq_prob = 1
    user = data_points['user'].to(config['device'])
    item = data_points['item'].to(config['device'])
    bos = data_points['seq'][:, 0].unsqueeze(0).to(config['device'])  # (1, batch_size)
    text = bos  # (src_len - 1, batch_size)
    # start_idx = text.size(0)

    for idx in range(config['seq_max_len']):
        # produce a word at each step
        if idx == 0:  # predict word from <bos>
            # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            log_word_prob, log_context_dis, rating_p, _ = model(user, item, text, False)
        else:
            log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)
        word_prob = log_word_prob.exp()  # (batch_size, ntoken)
        largest_word_prob, word_idx = torch.max(word_prob, dim=1)  # (batch_size,)
        text = torch.cat([text, word_idx.unsqueeze(0)], 0)  # (len++, batch_size)
        # seq_prob *= largest_word_prob.item()  # (1)->scalar
    ids = text.t().tolist()  # (batch_size, seq_len)
    for li in ids:
        li.append(2)

    return ids, seq_prob


def gen_seq_peter_topk(data_points, model, config):
    user = data_points['user'].to(config['device'])
    item = data_points['item'].to(config['device'])
    bos = data_points['seq'][:, 0].unsqueeze(0).to(config['device'])  # (1, batch_size)
    text = bos  # (src_len - 1, batch_size)
    # start_idx = text.size(0)

    for idx in range(config['seq_max_len']):
        # produce a word at each step
        if idx == 0:  # predict word from <bos>
            # (batch_size, ntoken) vs. (batch_size, ntoken) vs. (batch_size,)
            log_word_prob, log_context_dis, rating_p, _ = model(user, item, text, False)
        else:
            log_word_prob, _, _, _ = model(user, item, text, False, False, False)  # (batch_size, ntoken)
        word_prob = log_word_prob.exp()  # (batch_size, ntoken)
        word_idx = torch.multinomial(word_prob, num_samples=1, replacement=True)  # (batch_size, 1)
        text = torch.cat([text, word_idx.t()], 0)  # (len++, batch_size)
    ids = text.t().tolist()  # (batch_size, seq_len)
    for li in ids:
        li.append(2)

    return ids

