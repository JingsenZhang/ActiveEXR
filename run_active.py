import os
import torch
import argparse
import torch.nn as nn

from utils import now_time, set_seed, get_model, get_trainer_single
from data.dataloader import DataLoader
from data.batchify import Batchify
from config import Config
from active_learning.annotate_batch import get_annotations

###############################################################################
# Params
###############################################################################
parser = argparse.ArgumentParser(description='Active learning')
parser.add_argument('--model', '-m', type=str, default='PETER',
                    help='base model name')
parser.add_argument('--dataset', '-d', type=str, default='trip',
                    help='dataset name')
parser.add_argument('--config', '-c', type=str, default='peter.yaml',
                    help='config files')
args, _ = parser.parse_known_args()

config_file_list = args.config.strip().split(' ') if args.config else None
config = Config(config_file_list=config_file_list).final_config_dict
print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for param in config:
    print('{:40} {}'.format(param, config[param]))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

data_path = 'dataset/' + args.dataset + '/reviews.pickle'
index_dir = 'dataset/' + args.dataset + '/' + config['split'] + '/'

set_seed(config['seed'])
if torch.cuda.is_available():
    if not config['cuda']:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if config['cuda'] else 'cpu')
if config['cuda']:
    torch.cuda.set_device(config['gpu_id'])

if not os.path.exists(config['checkpoint']):
    os.makedirs(config['checkpoint'])
generated_file = args.model + '_' + args.dataset + '_' + config['method'] + '_' + config['unique'] + '_' + str(config['seed']) + '_'
prediction_path = os.path.join(config['checkpoint'], generated_file)

###############################################################################
# Load data
###############################################################################
print(now_time() + 'Loading data')
corpus = DataLoader(data_path, index_dir, config['vocab_size'])
word2idx = corpus.word_dict.word2idx
idx2word = corpus.word_dict.idx2word
pad_idx = word2idx['<pad>']
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntokens = len(corpus.word_dict)
nfeature = len(corpus.feature_set)
feature_set = list(corpus.feature_set)
trainset_size = corpus.train_size
validset_size = corpus.valid_size
testset_size = corpus.test_size
unlabel_size = corpus.unlabel_size
train_set = corpus.train
val_set = corpus.valid
unlabel_set = corpus.unlabel
unlabel_idxs = set([i for i in range(len(unlabel_set))])

train_data = Batchify(corpus.train, word2idx, trainset_size, config['seq_max_len'], config['batch_size'], shuffle=True)
val_data = Batchify(corpus.valid, word2idx, validset_size, config['seq_max_len'], config['batch_size'])
test_data = Batchify(corpus.test, word2idx, testset_size, config['seq_max_len'], config['batch_size'])
# just for rating pre-train
unlabel_data = Batchify(corpus.unlabel, word2idx, unlabel_size, config['seq_max_len'], config['batch_size'], shuffle=True)
print(now_time() + 'Dataset {}: nuser:{} | nitem:{} | ntoken:{} | nfeature:{} '.format(args.dataset, nuser, nitem, ntokens, nfeature))
print(now_time() + 'Initial trainset:{} | validset:{} | testset:{} | unlabel:{} '.format(trainset_size, validset_size, testset_size, unlabel_size))

###############################################################################
# Update params dict
###############################################################################
text_criterion = nn.NLLLoss(ignore_index=pad_idx, reduction='none')  # ignore the padding when computing loss
rating_criterion = nn.MSELoss(reduction='none')
feature_criterion = nn.CrossEntropyLoss(reduction='mean')
# all
config['nuser'] = nuser
config['nitem'] = nitem
config['ntoken'] = ntokens
config['device'] = device
config['word2idx'] = word2idx
config['idx2word'] = idx2word
config['text_criterion'] = text_criterion
config['rating_criterion'] = rating_criterion
config['prediction_path'] = prediction_path
# PETER
if config['use_feature']:
    src_len = 2 + train_data.feature.size(1)  # [u, i, f]
else:
    src_len = 2  # [u, i]
tgt_len = config['seq_max_len'] + 1  # added <bos> or <eos>
config['src_len'] = src_len
config['tgt_len'] = tgt_len
config['pad_idx'] = pad_idx


###############################################################################
# Build the model
###############################################################################
model = get_model(args.model)(config).to(device)
trainer = get_trainer_single(args.model)(config, model)

###############################################################################
# Active Learning
###############################################################################
# model pre-training
print('\n')
print(now_time() + '\033[1;36m' + 'Train model on seed data: ' + '\033[0m')
model_path = trainer.train_loop(train_data, val_data, model)
print(now_time() + 'Load the best model' + model_path)
model.load_state_dict(torch.load(model_path))
test_RMSE, test_MAE, BLEU1, BLEU4, ROUGE = trainer.test_loop(test_data, model, 1)

# active learning
print('\n')
print(now_time() + '\033[1;36m' + 'Start active learning: ' + '\033[0m')
tot_round = int(config['cost'] / config['select_num'])
for round in range(1, tot_round + 1):
    # annotate
    print('\n')
    print(now_time() + '\033[1;32m' + 'AC round {}: Step1: get annotations '.format(round) + '\033[0m')
    train_set, unlabel_idxs = get_annotations(unlabel_idxs, unlabel_set, train_set, train_data, val_data, model, config)
    print(now_time() + 'AC round {}: tot_annotated_num: {} | trainset:{} | unlabel_set: {} '.format(round,
                                                                                                    round * config['select_num'],
                                                                                                    len(train_set),
                                                                                                    len(unlabel_idxs)))
    # data update
    train_data = Batchify(train_set, word2idx, len(train_set), config['seq_max_len'], config['batch_size'],
                          shuffle=True)
    # re-train model
    print(now_time() + '\033[1;33m' + 'AC round {}: Step2: model training & testing'.format(round) + '\033[0m')

    model_path = trainer.train_loop(train_data, val_data, model)
    print(now_time() + 'Load the best model' + model_path)
    model.load_state_dict(torch.load(model_path))
    test_RMSE, test_MAE, BLEU1, BLEU4, ROUGE = trainer.test_loop(test_data, model, round % 5 == 0 or round == 1)

    if len(unlabel_idxs) == 0:
        print(now_time() + 'All samples have been labeled.')
        break


print('=' * 89)
print(now_time() + 'Finally run on test data:')
print(now_time() + 'Load the best model' + model_path)
model.load_state_dict(torch.load(model_path))
test_RMSE, test_MAE, BLEU1, BLEU4, ROUGE = trainer.test_loop(test_data, model, 1)
print('Final test: RMSE {:7.4f} | MAE {:7.4f}'.format(test_RMSE, test_MAE))
print('Final test: BLEU-1 {:7.4f} | BLEU-4 {:7.4f}'.format(BLEU1, BLEU4))
for (k, v) in ROUGE.items():
    print('Final test: {} {:7.4f}'.format(k, v))
print(now_time() + '\033[1;31m' + '{} on {} dataset: Active learning is OK!'.format(args.model, args.dataset) + '\033[0m')
print('=' * 89)
