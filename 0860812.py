from torch.utils.data import WeightedRandomSampler

from datasets import CIFAR10_truncated
import argparse
from utils import progress_bar
from cifar10_models import *

from torchvision import transforms
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', type=str, required=False, default="./data/cifar10", help="Data directory")
parser.add_argument('--train_bs', default=256, type=int, help='training batch size')
parser.add_argument('--test_bs', default=100, type=int, help='testing batch size')
parser.add_argument('--lr_schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--reweight_balance', type=bool, default=False,
                    help='Enable reweight balance data (default:False)')
parser.add_argument('--resampling_balance', type=bool, default=False,
                    help='Enable resampling balance data (default:False)')
args = parser.parse_args()

'''Function for sampling image according to each labels count'''

def resampling_balance(data):
    targets = data.target
    class_count = np.unique(targets, return_counts=True)[1]
    print("Class number before resampling: ",class_count)

    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_dataloader(datadir, train_bs, test_bs, dataidxs=None):
    # transform = transforms.Compose([transforms.ToTensor()])

    transform = transforms.Compose([

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    train_ds = CIFAR10_truncated(datadir, dataidxs=dataidxs, train=True, transform=transform, download=True)
    test_ds = CIFAR10_truncated(datadir, train=False, transform=transform, download=True)

    if args.resampling_balance:
        # Oversample the minority classes
        train_sampler = resampling_balance(train_ds)
        train_dl = data.DataLoader(dataset=train_ds, sampler=train_sampler, batch_size=train_bs, num_workers=2)
        test_sampler = resampling_balance(test_ds)
        test_dl = data.DataLoader(dataset=test_ds, sampler=test_sampler, batch_size=test_bs, shuffle=False,
                                  num_workers=2)


        print("Resampling Works!")
        targets= train_dl.dataset.target
        class_count = np.unique(targets, return_counts=True)[1]
        print("Data training after resample:", class_count)

        targets_test= test_dl.dataset.target
        class_count = np.unique(targets_test, return_counts=True)[1]
        print("Data test after resample:", class_count)
    else:
        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, num_workers=2)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=2)
        print("Resampling Turned Off!")

    return train_dl, test_dl


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch in args.lr_schedule:
        lr *= args.lr_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def countX(tup, x):
    count = 0
    for ele in tup:
        if (ele == x):
            count = count + 1
    return count


if __name__ == '__main__':
    dataidxs = []
    # load the index of imbalanced CIFAR-10 from dataidx.txt
    with open("dataidx.txt", "r") as f:
        for line in f:
            dataidxs.append(int(line.strip()))
    # get the training/testing data loader
    train_dl, test_dl = get_dataloader(args.datadir, args.train_bs, args.test_bs, dataidxs)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch


    net = densenet161(pretrained=True)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    lr = 0.1

    '''Apply the Weight-Balanced Loss'''

    if args.reweight_balance:
        '''Get the label of the dataset'''
        print("Reweighting loss works!")
        target = train_dl.dataset.target

        dicts_target = {}

        '''Count each label to determine weight later'''
        for a in range(0, 10):
            dicts_target[a] = countX(target, a)

        beta = (sum(dicts_target.values()) - 1) / sum(dicts_target.values())
        print("Beta is ",beta)
        '''Count the weight according to the number of label (Class Balanced Loss) '''
        weights = []
        for key, value in dicts_target.items():
            weight = (1 - beta) / (1 - (beta ** value))
            weights.append(weight)

        class_weights = torch.FloatTensor(weights).cuda()
        print("Weights according to imbalanced dataset", weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("Reweighting loss turned off!")

        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_dl), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return train_loss / (batch_idx + 1), 100. * correct / total


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_dl):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets.long())

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(test_dl), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

        return test_loss / (batch_idx + 1), 100. * correct / total


    def confMatrix():
        nb_classes = 10

        PATH="checkpoint/ckpt.pth"
        checkpoint=torch.load(PATH)
        net.load_state_dict(checkpoint["net"])

        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_dl):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)
                _, preds = outputs.max(1)
                for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        return confusion_matrix / confusion_matrix.sum(1)


    training_loss = []
    testing_loss = []
    training_accuracy = []
    testing_accuracy = []

    for epoch in range(start_epoch, start_epoch +75):
        adjust_learning_rate(optimizer, epoch, lr)
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)

        training_loss.append(train_loss)
        training_accuracy.append(train_acc)

        testing_loss.append(test_loss)
        testing_accuracy.append(test_acc)



    confmatrix = confMatrix()
    sn.heatmap(confmatrix, annot=True)  # font size
    plt.show()

    plt.plot(training_accuracy)
    plt.plot(testing_accuracy)
    plt.title('Plot of overall training accuracy vs test acuracy')
    plt.xlim([0, 75])
    plt.ylim([0, 100])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['x_train', 'x_test'], loc='upper right')
    plt.show()
