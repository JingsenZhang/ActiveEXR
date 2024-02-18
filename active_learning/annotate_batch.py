import random
import torch
from data import Batchify
from active_learning.influence import IF


def calculate_entropy(model, data, config):
    model.eval()
    candidate_voi = torch.tensor([]).to(config['device'])
    with torch.no_grad():
        while True:
            user, item, rating, seq, feature, _, _ = data.next_batch()  # (batch_size, seq_len), data.step += 1
            user = user.to(config['device'])  # (batch_size,)
            item = item.to(config['device'])
            seq = seq.to(config['device'])  # (batch_size, seq_len + 2)
            text = seq.t()[:-1]  # (src_len + tgt_len - 2, batch_size)
            log_word_prob, _, _, _ = model(user, item, text)  # (tgt_len, batch_size, ntoken)
            log_word_prob = torch.transpose(log_word_prob, 0, 1)  # (batch_size, tgt_len, ntoken)
            # multiply each probability by its base 2 log
            prob_dist = log_word_prob.exp()  # (batch_size, tgt_len, ntokens)
            log_probs = prob_dist * torch.log2(prob_dist + config['protect'])  # (batch_size, tgt_len, ntoken)  # prob_dist+1e-15
            # (batch_size, tgt_len, ntoken) -> (batch_size, tgt_len*ntoken)
            log_probs = log_probs.reshape(log_probs.size(0), -1)
            # (batch_size, tgt_len*ntoken) -> (batch_size,)
            raw_entropy = 0 - torch.sum(log_probs, dim=1)
            # normalized_entropy = raw_entropy / math.log2(log_probs[0].numel())
            candidate_voi = torch.concat([candidate_voi, raw_entropy], dim=0)

            if data.step == data.total_step:
                break

    return candidate_voi  # (data.len)


def calculate_influence(model, reference_data, train_data, candidate_data, config):
    influ = IF(model, reference_data, train_data, candidate_data, config)
    if config['influ_bs']:
        candidate_voi = influ.get_influence_function_batch()
    else:
        candidate_voi = influ.get_influence_function()
    return candidate_voi


def calculate_both(model, reference_data, train_data, candidate_data, config):
    entropy = calculate_entropy(model, candidate_data, config)
    normal_entropy = entropy / torch.norm(entropy, 2)
    influence = calculate_influence(model, reference_data, train_data, candidate_data, config)
    normal_influence = influence / torch.norm(influence, 2)
    return config['en_weight'] * normal_entropy + config['in_weight'] * normal_influence


def get_annotations(unlabel_idxs, unlabel_set, train_set, train_data, reference_data, model, config):
    print('select method: ', config['method'])
    if len(unlabel_idxs) > config['can_num']:
        candidate_idxs = random.sample(unlabel_idxs, config['can_num'])
    else:
        candidate_idxs = unlabel_idxs
    candidate_set = [unlabel_set[idx] for idx in candidate_idxs]

    if len(candidate_idxs) > config['select_num']:
        candidate_data = Batchify(candidate_set, config['word2idx'], len(candidate_set), config['seq_max_len'], config['influ_bs'], shuffle=False)
        candidate_voi = None
        if config['method'] == 'en':
            candidate_voi = calculate_entropy(model, candidate_data, config)
        elif config['method'] == 'in':
            candidate_voi = calculate_influence(model, reference_data, train_data, candidate_data, config)
        elif config['method'] == 'bo':
            candidate_voi = calculate_both(model, reference_data, train_data, candidate_data, config)
        voi_rank = torch.argsort(candidate_voi, descending=True)
        selected_idxs = [candidate_idxs[idx] for idx in voi_rank[:config['select_num']]]
    else:
        selected_idxs = candidate_idxs

    for idx in selected_idxs:
        train_set.append(unlabel_set[idx])
        unlabel_idxs.remove(idx)

    return train_set, unlabel_idxs
