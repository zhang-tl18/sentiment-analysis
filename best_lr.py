import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
import os
import matplotlib.pyplot as plt

from dataset import My_Dataset
import models

# parameters
parser = argparse.ArgumentParser(description='Sentiment Analysis') 
parser.add_argument('--model', type=str, default='MLP', help='Model Name')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
options = parser.parse_args()
print(options)

train_file = './Dataset/train.txt'
valid_file = './Dataset/validation.txt'
test_file = './Dataset/test.txt'
word2vec_file = './Dataset/wiki_word2vec_50.bin'
img_path = './img/lr-acc/'
n_epochs = options.epochs
batch_size_trian = 256
batch_size_valid = 512
random_seed = 2023
learning_rate = 0.001
max_lr = 0.0016      # 0.008
base_lr = 0.001     # 0.0016
step_size = 10


torch.manual_seed(random_seed)


# data loader
print('\nLoading data...')
train_set = My_Dataset(train_file, word2vec_file)
valid_set = My_Dataset(valid_file, word2vec_file)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size_trian, shuffle=True, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size_valid, shuffle=True, pin_memory=True)
print('Data loaded!\n')


# build model
network = getattr(models, options.model)()
if torch.cuda.is_available():
    network.cuda()
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, step_size_down=step_size, cycle_momentum=False)
criterion = nn.CrossEntropyLoss()


# training process
acc_list = []
lr_list = []

def train(epoch):
    network.train()
    print('Train Epoch', epoch)
    for batch_idx, (data, target, input_len) in enumerate(train_loader):
        # calculate
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = network(data, input_len)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()


def test_acc(data_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, input_len in data_loader:
            # calculate
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = network(data, input_len)
            loss = criterion(output, target).item()
            pred = output.detach().max(1)[1]
            correct += (target == pred.view_as(target)).sum().item()
            # record
            test_loss += loss
        test_loss /= len(data_loader)
    return correct, len(data_loader.dataset), test_loss


def test(epoch):
    global loss_min
    global stop_count
    # valid accuracy
    correct, total, loss = test_acc(valid_loader)
    acc_list.append(correct/total)
    lr_list.append(optimizer.param_groups[0]['lr'])
    print("lr: ", optimizer.param_groups[0]['lr'])


def plot_figure():
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    fig = plt.figure()
    plt.plot(lr_list, acc_list, color='blue')
    plt.xlabel('lr')
    plt.ylabel('acc')
    plt.savefig(img_path + network.name + ' ' + str(len(os.listdir(img_path))) + '.png')
    plt.show()


if __name__ == "__main__":
    for epoch in range(1, n_epochs + 1):
        print("-" * 40)
        train(epoch)
        test(epoch)
    print("-" * 40)

    plot_figure()
