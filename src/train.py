from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
from datetime import datetime
from random import shuffle
from tqdm import trange
import time
import argparse
import pickle, json
import gensim

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('./')

from src.utils import print_config, save_checkpoint, save_embedding, construct_feature
from src.models import GCNDecoder, MLP
from src.dataset import Dataset, EvaDataset
from src.logger import myLogger


def parse_args():
    # general settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dblp-sub',
                        help='dataset name. [dblp, dblp-sub]')
    parser.add_argument('--eval_file', type=str, default='',
                        help='evaluation file path.')
    parser.add_argument("--load_model", type=str, default=False,
                        help="whether to load model")
    parser.add_argument("--budget", type=int, default=50,
                        help="budget for greedy search")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use")
    parser.add_argument('--log_level', default=20,
                        help='logger level.')
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix use as addition directory")
    parser.add_argument('--suffix', default='', type=str,
                        help='suffix append to log dir')
    parser.add_argument('--log_every', type=int, default=100,
                        help='log results every epoch.')
    parser.add_argument('--save_every', type=int, default=100,
                        help='save learned embedding every epoch.')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')


    # sample settings
    parser.add_argument("--sample_mode", type=str, default='n|l|dc|ds',
                        help="n:node, l:link, dc:diffusion content, ds:diffusion structure")
    parser.add_argument('--diffusion_threshold', default=10, type=int,
                        help='threshold for diffusion')
    parser.add_argument('--neighbor_sample_size', default=30, type=int,
                        help='sample size for neighbor to be used in gcn')
    parser.add_argument('--sample_size', default=200, type=int,
                        help='sample size for training data')
    parser.add_argument('--negative_sample_size', default=1, type=int,
                        help='negative sample / positive sample')
    parser.add_argument('--sample_embed', default=100, type=int,
                        help='sample size for embedding generation')


    # training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=100,
                        help='Number of hidden units, also the dimension of node representation after GCN.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--relation', type=int, default=2,
                        help='Number of relations.')
    parser.add_argument('--rel_dim', type=int, default=300,
                        help='dimension of relation embedding.')
    parser.add_argument('--hard_gumbel', type=int, default=0,
                        help='whether to make gumbel one hot.')
    parser.add_argument('--use_superv', type=int, default=1,
                        help='whether to add supervision(prior) to variational inference')
    parser.add_argument('--superv_ratio', type=float, default=0.8,
                        help='how many data to use in training(<=0.8), default is 80%, note that 20% are used as test')
    parser.add_argument('--t', type=float, default=0.4,
                        help='ratio of supervision in sampled data')
    parser.add_argument('--a', type=float, default=0.25,
                        help='weight for node feature loss')
    parser.add_argument('--b', type=float, default=0.25,
                        help='weight for link loss')
    parser.add_argument('--c', type=float, default=0.25,
                        help='weight for diffusion loss')
    parser.add_argument('--d', type=float, default=0.25,
                        help='weight for diffusion content loss')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='temperature for gumbel softmax')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='whether to use early stop')
    parser.add_argument('--patience', type=int, default=500,
                        help='used for early stop')

    # evluating settings
    parser.add_argument('--epochs_eval', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr_eval', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--batch_size_eval', type=float, default=10,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_eval', type=int, default=100,
                        help='Number of hidden units.')
    parser.add_argument('-patience_eval', type=int, default=500,
                        help='used for early stop in evaluation')

    return parser.parse_args()


def evaluate(args, embedding, logger, repeat_times=5):
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []
    if args.use_superv:
        train = construct_feature(args.train, embedding)
        test = construct_feature(args.test, embedding)
    else:
        data = construct_feature(args.label_data, embedding)
        split = int(len(args.label_data) / repeat_times)

    for i in range(repeat_times):
        if not args.use_superv:
            p1, p2 = i*split, (i+1)*split
            test = data[p1:p2, :]
            train1, train2 = data[:p1, :], data[p2:, :]
            train = np.concatenate([train1, train2])

        X_train, y_train = torch.FloatTensor(train[:, :-1]), torch.LongTensor(train[:, -1])
        X_test, y_test = torch.FloatTensor(test[:, :-1]), torch.LongTensor(test[:, -1])
        dataloader = DataLoader(EvaDataset(X_train, y_train), batch_size=args.batch_size_eval, shuffle=True)
        X_train = X_train.to(args.device)
        X_test = X_test.to(args.device)
        y_train = y_train.to(args.device)
        y_test = y_test.to(args.device)

        kwargs = {
            'input_dim': X_train.size(1),
            'hidden_dim': args.hidden_eval,
            'output_dim': args.output_dim
        }
        model = MLP(**kwargs).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_eval)
        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        count = 0
        for epoch in range(args.epochs_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch.to(args.device), label.to(args.device))
                loss.backward()
                optimizer.step()

            preds, test_acc = model.predict(X_test, y_test)
            test_acc *= 100
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch + 1
                best_pred = preds
                count = 0
            else:
                count += 1
                if count >= args.patience_eval:
                    break

            _, train_acc = model.predict(X_train, y_train)
            train_acc *= 100
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1

            print('\repoch {}/{} train acc={:.4f}, test acc={:.4f}, best train acc={:.4f} @epoch:{:d}, best test acc={:.4f} @epoch:{:d}'.
                      format(epoch + 1, args.epochs_eval, train_acc, test_acc, best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch), end='')
            sys.stdout.flush()

        print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs)
    std = np.std(best_test_accs)
    logger.info('{}: best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f} @epoch:{:d}'.
                format(args.eval_file, best_train_acc, int(best_train_acc_epoch), best_test_acc, std, int(best_test_acc_epoch)))

    return best_train_acc, best_test_acc, std


def train(args, embedding, Data, log_dir, logger, writer=None):
    # Model and optimizer
    model = GCNDecoder(device=args.device,
                        embedding=embedding,
                       nfeat=args.feature_len,
                       nhid=args.hidden,
                       ncont=args.content_len,
                       nrel=int(args.relation), rel_dim=args.rel_dim,
                       dropout=args.dropout,
                       a=float(args.a), b=float(args.b), c=float(args.c), d=float(args.d),
                       tau=args.tau, hard_gumbel=args.hard_gumbel)
    model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.modes = []
    if 'n' in args.sample_mode and model.a > 0:
        args.modes.append('node')
    if 'l' in args.sample_mode and model.b > 0:
        args.modes.append('link')
    if 'dc' in args.sample_mode and model.d > 0:
        args.modes.append('diffusion_content')
    if 'ds' in args.sample_mode and model.c > 0:
        args.modes.append('diffusion_structure')

    t = time.time()
    best_acc, best_epoch, best_std = 0, 0, -1
    count = 0
    model.train()

    for epoch in range(1, args.epochs+1):
        losses = []
        for mode in args.modes:
            optimizer.zero_grad()
            (sampled_features, sampled_adj, prior, nodes), sampled_labels = Data.sample(mode, return_nodes=True)
            loss = model(sampled_features, sampled_adj, sampled_labels, mode, nodes, prior)
            if torch.isnan(loss):
                print('nan loss!')
            loss.backward()
            optimizer.step()

            if epoch % args.log_every == 0:
                losses.append(loss.item())

        if epoch % args.log_every == 0:
            duration = time.time() - t
            msg = 'Epoch: {:04d} '.format(epoch)
            for mode, loss in zip(args.modes, losses):
                msg += 'loss_{}: {:.4f}\t'.format(mode, loss)
                if writer is not None:
                    writer.add_scalar('data/{}_loss'.format(mode), loss, epoch)
            logger.info(msg+' time: {:d}s '.format(int(duration)))

        if epoch % args.save_every == 0:
            learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(Data.feature_len)
            embedding = model.embedding(torch.LongTensor(args.nodes).to(args.device)).data.cpu().numpy()
            learned_embed.add([str(node) for node in args.nodes], embedding)
            train_acc, test_acc, std = evaluate(args, learned_embed, logger)
            duration = time.time() - t
            logger.info('Epoch: {:04d} '.format(epoch)+
                        'train_acc: {:.2f} '.format(train_acc)+
                        'test_acc: {:.2f} '.format(test_acc)+
                        'std: {:.2f} '.format(std)+
                        'time: {:d}s'.format(int(duration)))
            if writer is not  None:
                writer.add_scalar('data/test_acc', test_acc, epoch)
            if test_acc > best_acc:
                best_acc = test_acc
                best_std = std
                best_epoch = epoch
                save_embedding(learned_embed, os.path.join(log_dir, 'embedding.bin'))
                save_checkpoint({
                    'args': args,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, log_dir,
                    f'epoch{epoch}_time{int(duration):d}_trainacc{train_acc:.2f}_testacc{test_acc:.2f}_std{std:.2f}.pth.tar', logger, True)
                count = 0
            else:
                if args.early_stop:
                    count += args.save_every
                if count >= args.patience:
                    logger.info('early stopped!')
                    break

    logger.info(f'best test acc={best_acc:.2f} +- {best_std:2f} @ epoch:{int(best_epoch):d}')
    return best_acc


if __name__ == '__main__':
    # Initialize args and seed
    args = parse_args()
    # print('Number CUDA Devices:', torch.cuda.device_count())
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    # torch.cuda.device(args.gpu)
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else "cpu")
    # print('Active CUDA Device: GPU', torch.cuda.current_device())
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    args.superv_ratio = float(args.superv_ratio)

    # Load data
    if not args.eval_file:
        args.eval_file = f'data/{args.dataset}/eval/rel.txt'
    labels, labeled_data = set(), []
    nodes = set()
    with open(args.eval_file, 'r') as lf:
        for line in lf:
            if line.rstrip() == 'test':
                continue
            line = line.rstrip().split('\t')
            data1, data2, label = line[0], line[1], int(line[2])
            labeled_data.append((data1, data2, label))
            labels.add(label)
            nodes.update([int(data1), int(data2)])
    shuffle(labeled_data)
    args.nodes = list(nodes)
    args.label_data = labeled_data
    args.output_dim = len(labels)

    if args.use_superv:
        args.relation = len(labels)
        test_split = int(0.8*len(args.label_data))
        args.test = args.label_data[test_split:]
        train_split = int(args.superv_ratio*len(args.label_data))
        args.train = args.label_data[:train_split]

    Data = Dataset(args, args.dataset)
    args.feature_len = Data.feature_len
    args.content_len = Data.content_len
    args.num_node, args.num_link, args.num_diffusion = Data.num_node, Data.num_link, Data.num_diff

    # initialize logger
    comment = f'_{args.dataset}_{args.suffix}'
    current_time = datetime.now().strftime('%b_%d_%H-%M-%S')
    if args.prefix:
        base = os.path.join('running_log', args.prefix)
        log_dir = os.path.join(base, args.suffix)
    else:
        log_dir = os.path.join('running_log', current_time + comment)
    args.log_dir = log_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = myLogger(name='exp', log_path=os.path.join(log_dir, 'log.txt'))
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Parameters', str(vars(args)))
    print_config(args, logger)
    logger.setLevel(args.log_level)

    # start
    # initial embedding by aggregating feature
    embedding = torch.FloatTensor(Data.num_node, Data.feature_len)
    all_nodes = list(range(Data.num_node))
    t = time.time()
    for i in trange(0, Data.num_node, args.sample_embed):
        nodes = all_nodes[i:i + args.sample_embed]
        features, adj, _ = Data.sample_subgraph(nodes, False)
        features = features.to(args.device)
        adj = adj.to(args.device)
        embedding[i:i + len(nodes)] = torch.spmm(adj, features)
    duration = time.time() - t
    logger.info(f'initialize embedding time {int(duration):d}')

    # Train model
    t_total = time.time()
    train(args, embedding, Data, args.log_dir, logger, writer)
    logger.info("Optimization Finished!")
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
