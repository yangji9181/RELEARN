import os, sys
import os.path as osp
import argparse
import gensim
import numpy as np
from random import shuffle
from tqdm import tqdm, trange
from collections import defaultdict, namedtuple

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.enabled = True


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True),nn.ReLU(inplace=True), nn.Linear(hidden_dim, output_dim, bias=True))
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, label):
        return self.loss_fn(self.classifier(batch), label)

    def predict(self, batch, label):
        self.eval()
        _, predicted = torch.max(self.classifier(batch), 1)
        c = (predicted == label).squeeze()
        precision = torch.sum(c).item() / float(c.size(0))
        self.train()
        return predicted.cpu().numpy(), precision


class EvaDataset(Dataset):
    def __init__(self, X, y):
        self.len = len(X)
        self.data = [(X[i], y[i]) for i in range(self.len)]

    def __getitem__(self, index):
        batch, label = self.data[index]
        return batch, label

    def __len__(self):
        return self.len


def parse_args():
    parser = argparse.ArgumentParser()
    # general para
    parser.add_argument('-g', '--gpu', type=int, default=-1, help='gpu id')
    parser.add_argument('-d', '--label_path', type=str, default='data/rel1.txt', help='data path')
    parser.add_argument('-m', '--model_path', type=str, default='baseline/baseline.sub.line.txt', help='model path')
    parser.add_argument('-o', '--output_path', type=str, default='', help='output path')

    # general optimization para
    parser.add_argument("--method", type=str, default='LR', help='evaluation method', choices=['MLP', 'SVM', 'LR'])
    parser.add_argument('-e', '--num_epoch', type=int, default=100, help='epoch number')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    # MLP para
    parser.add_argument('--hidden_layer_dim', type=int, default=100, help='hidden layer dimension')
    return parser.parse_args()


def node_classification(args, embedding, file, repeat_times=10):
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []
    labels, labeled_data = set(), []
    with open(file, 'r') as lf:
        for line in lf:
            if line.rstrip() == 'test':
                # finish load training data
                train, _, _ = construct_feature(labeled_data, embedding)
                labeled_data = []
                continue
            line = line.rstrip().split('\t')
            data1, data2, label = line[0], line[1], int(line[2])
            labeled_data.append((data1, data2, label))
            labels.add(label)
    output_dim = len(labels)

    test, _, _ = construct_feature(labeled_data, embedding)

    if len(non_ex_wrd) > 0:
        print('found {} words not in embedding'.format(len(non_ex_wrd)))

    if args.method == 'MLP':
        for i in range(repeat_times):
            np.random.shuffle(train)

            X_train, y_train = torch.FloatTensor(train[:, :-1]), torch.LongTensor(train[:, -1])
            X_test, y_test = torch.FloatTensor(test[:,:-1]), torch.LongTensor(test[:,-1])
            dataloader = DataLoader(EvaDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
            X_train = X_train.to(args.device)
            X_test = X_test.to(args.device)
            y_train = y_train.to(args.device)
            y_test = y_test.to(args.device)

            kwargs = {
                'input_dim': X_train.size(1),
                'hidden_dim': args.hidden_layer_dim,
                'output_dim': output_dim
            }
            model = MLP(**kwargs).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            best_test_acc, best_train_acc = 0, 0
            best_test_acc_epoch, best_train_acc_epoch = 0, 0
            for epoch in range(args.num_epoch):
                for i, (batch, label) in enumerate(dataloader):
                    optimizer.zero_grad()
                    loss = model(batch.to(args.device), label.to(args.device))
                    loss.backward()
                    optimizer.step()

                preds, test_acc = model.predict(X_test, y_test)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_test_acc_epoch = epoch + 1
                    best_pred = preds

                _, train_acc = model.predict(X_train, y_train)
                if train_acc > best_train_acc:
                    best_train_acc = train_acc
                    best_train_acc_epoch = epoch + 1

                print('\repoch {}/{} train acc={}, test acc={}, best train acc={} @epoch:{}, best test acc={} @epoch:{}'.
                      format(epoch + 1, args.num_epoch, train_acc, test_acc, best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch), end='')
                sys.stdout.flush()

            print('')
            best_train_accs.append(best_train_acc)
            best_test_accs.append(best_test_acc)
            best_train_acc_epochs.append(best_train_acc_epoch)
            best_test_acc_epochs.append(best_test_acc_epoch)

        best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
            np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs)
        std = np.std(best_test_accs)
        print('{}: best train acc={} @epoch:{}, best test acc={} += {} @epoch:{}'.
              format(file, best_train_acc, best_train_acc_epoch, best_test_acc, std, best_test_acc_epoch))

    elif args.method == 'LR':
        for i in range(repeat_times):
            np.random.shuffle(train)
            X = train[:, :-1]
            y = train[:, -1]
            clf = LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X, y)
            train_pred = clf.predict(X)
            test_pred = clf.predict(test[:, :-1])
            best_train_acc = sum(train_pred==y) / len(y)
            best_test_acc = sum(test_pred==test[:, -1]) / len(test_pred)
            best_train_accs.append(best_train_acc)
            best_test_accs.append(best_test_acc)
            print('{}: train acc={}, test acc={}'.format(i, best_train_acc, best_test_acc))
            del clf

        best_train_acc, best_test_acc, std = np.mean(best_train_accs), np.mean(best_test_accs), np.std(best_test_accs)
        print('{}: best train acc={}, best test acc={} += {}'.format(file, best_train_acc, best_test_acc, std))

    elif args.method == 'SVM':
        for i in range(repeat_times):
            np.random.shuffle(train)
            X = train[:, :-1]
            y = train[:, -1]
            clf = SVC().fit(X, y)
            train_pred = clf.predict(X)
            test_pred = clf.predict(test[:, :-1])
            best_train_acc = sum(train_pred==y) / len(y)
            best_test_acc = sum(test_pred==test[:, -1]) / len(test_pred)
            best_train_accs.append(best_train_acc)
            best_test_accs.append(best_test_acc)
            print('{}: train acc={}, test acc={}'.format(i, best_train_acc, best_test_acc))

        best_train_acc, best_test_acc, std = np.mean(best_train_accs), np.mean(best_test_accs), np.std(best_test_accs)
        print('{}: best train acc={}, best test acc={} += {}'.format(file, best_train_acc, best_test_acc, std))

    return best_train_acc, best_test_acc, std


non_ex_wrd = set()


def construct_feature(data, w2v):
    global non_ex_wrd
    Data, labels, idx2word = [], [], []
    idx = 0
    for word1, word2, label in data:
        try:
            vector1 = w2v[word1]
        except:
            non_ex_wrd.add(word1)
            continue
        try:
            vector2 = w2v[word2]
        except:
            non_ex_wrd.add(word2)
            continue

        Data.append(np.concatenate([vector1, vector2]))
        labels.append(label)

        if word1 not in idx2word:
            idx2word.append(word1)
            idx += 1
        if word2 not in idx2word:
            idx2word.append(word2)
            idx += 1

    Data = np.concatenate((np.array(Data), np.array(labels)[:, np.newaxis]), axis=1)
    return Data, idx2word, non_ex_wrd


def compute_prec_recall(y_pred, y_true):
    label_stat = {i:[0,0, 0] for i in np.unique(y_true)}
    Stat = namedtuple('Stat', 'label precision recall')
    for i, pred in enumerate(y_pred):
        label_stat[y_true[i]][0] += pred == y_true[i]
        label_stat[y_true[i]][1] +=1
        label_stat[pred][2] += 1
    for label, (correct, s1, s2) in label_stat.items():
        if correct == 0:
            label_stat[label] = Stat(label=label, precision=0, recall=0)
        else:
            label_stat[label] = Stat(label=label, precision=float(correct)/s2, recall=float(correct)/s1)
    return label_stat


def repeated_evaluate(args):

    if osp.isfile(args.model_path):
        print('='*100)
        print('evaluating model saved in ', args.model_path)
        if not args.output_path:
            args.output_path = args.model_path + '.eva.tsv'
        print('saving results into', args.output_path)

        w2v = gensim.models.KeyedVectors.load_word2vec_format(args.model_path, binary=False)

        with open(args.output_path, 'w') as of:
            of.write("model file:\t{}\n".format(args.model_path))
            of.write('task file\tbest train acc\tbest test acc\tbest test std\n')

            if osp.isfile(args.label_path):
                task_file = args.label_path
                best_train_acc, best_test_acc, best_test_std = node_classification(args, embedding=w2v, file=task_file)
                of.write('{}\t{}\t{}\t{}\n'.format(task_file, best_train_acc, best_test_acc, best_test_std))
            else:
                for file in os.listdir(args.label_path):
                    task_file = osp.join(args.label_path, file)
                    best_train_acc, best_test_acc, best_test_std = node_classification(args, embedding=w2v, file=task_file)
                    of.write('{}\t{}\t{}\t{}\n'.format(task_file, best_train_acc, best_test_acc, best_test_std))

    elif osp.isdir(args.model_path):
        print('=' * 100)
        print('evaluating model saved in ', args.model_path)
        if not args.output_path:
            args.output_path = osp.join(args.model_path, 'eva.tsv')
        print('saving results into', args.output_path)

        with open(args.output_path, 'w') as of:
            of.write('model file\ttask file\tbest train acc\tbest test acc\tbest test std\n')

            task_accs = defaultdict(list)
            for file in os.listdir(args.model_path):
                file = osp.join(args.model_path, file)
                try :
                    w2v = gensim.models.KeyedVectors.load_word2vec_format(file, binary=False)
                except ValueError as e:
                    # print('{} may not be a w2v file'.format(file))
                    continue
                print('='*100)
                print('evaluating model saved in ', file)

                if osp.isfile(args.label_path):
                    task_file = args.label_path
                    best_train_acc, best_test_acc, best_test_std = node_classification(args, embedding=w2v, file=task_file)
                    task_accs[task_file].append(best_test_acc)
                    of.write('{}\t{}\t{}\t{}\n'.format(task_file, best_train_acc, best_test_acc, best_test_std))
                else:
                    for file in os.listdir(args.label_path):
                        task_file = osp.join(args.label_path, file)
                        best_train_acc, best_test_acc, best_test_std = node_classification(args, embedding=w2v, file=task_file)
                        task_accs[task_file].append(best_test_acc)
                        of.write('{}\t{}\t{}\t{}\n'.format(task_file, best_train_acc, best_test_acc, best_test_std))


if __name__ == '__main__':
    args = parse_args()
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else "cpu")
    repeated_evaluate(args)
