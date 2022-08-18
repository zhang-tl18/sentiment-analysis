import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
import time
import os
import matplotlib.pyplot as plt

from dataset import My_Dataset
import models

# parameters
parser = argparse.ArgumentParser(description='Sentiment Analysis') 
parser.add_argument('--model', type=str, default='TextCNN', help='Model Name')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--train', type=bool, default=True, help='Train or not')
parser.add_argument('--test', type=bool, default=False, help='Only test or not')
parser.add_argument('--continue', type=bool, default=False, help='Continue on the last training model or not', dest='continuing')
parser.add_argument('--regularization', type=float, default=1e-3, help='lambda in regularization')
parser.add_argument('--stop_count', type=int, default=5, help='early stop: epoch num can stand with no improvement')
parser.add_argument('--stop_delta', type=float, default=0.0015, help='early stop: delta when compare accuracy')
options = parser.parse_args()
print(options)

train_file = './Dataset/train.txt'
valid_file = './Dataset/validation.txt'
test_file = './Dataset/test.txt'
word2vec_file = './Dataset/wiki_word2vec_50.bin'
model_path = './Model/'
n_epochs = options.epochs
batch_size_trian = 128
batch_size_valid = 512
batch_size_test = 512
random_seed = 2023
record_size = 1024
learning_rate = 0.01
momentum = 0.5
lr_lamb = 0.9
max_stop_count = options.stop_count
stop_delta = options.stop_delta

torch.manual_seed(random_seed)


# data loader
print('\nLoading data...')
train_set = My_Dataset(train_file, word2vec_file)
valid_set = My_Dataset(valid_file, word2vec_file)
test_set = My_Dataset(test_file, word2vec_file)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size_trian, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size_valid, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=True)
print('Data loaded!\n')


# build model
network = getattr(models, options.model)()
if torch.cuda.is_available():
    network.cuda()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=options.regularization)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_lamb**epoch)
criterion = nn.CrossEntropyLoss()


# training process
train_losses = []
train_counter = []
valid_losses = []
valid_acc = []
valid_counter = []
acc_max = 0
stop_count = 0

def train(epoch):
    network.train()
    train_loss = 0
    record_loss = 0
    record_batch = 0
    print('Train Epoch', epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        # calculate
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #record
        train_loss += loss.item()
        record_loss += loss.item()
        record_batch += 1
        if batch_idx*batch_size_trian % record_size == 0:
            record_loss /= record_batch
            train_losses.append(record_loss)
            train_counter.append((batch_idx * batch_size_trian) + ((epoch - 1) * len(train_loader.dataset)))
            record_loss = 0
            record_batch = 0
    train_loss /= len(train_loader)
    print('train loss: {:.4f}'.format(train_loss))
    scheduler.step()


def test_acc(data_loader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            # calculate
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = network(data)
            loss = criterion(output, target).item()
            pred = output.detach().max(1)[1]
            correct += (target == pred.view_as(target)).sum().item()
            # record
            test_loss += loss
        test_loss /= len(data_loader)
    return correct, len(data_loader.dataset), test_loss


def test(epoch):
    global acc_max
    global stop_count
    # print message
    correct, total, loss = test_acc(train_loader)
    print('train accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100.*correct/total))
    correct, total, loss = test_acc(valid_loader)
    print('valid loss: {:.4f}'.format(loss))
    print('valid accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100.*correct/total))
    print('best  accuracy: ({:.2f}%)'.format(100 * max(correct / total, acc_max)))

    # record
    acc = correct / total
    valid_losses.append(loss)
    valid_acc.append(acc)
    valid_counter.append(epoch * len(train_loader.dataset))
    
    # early stop
    if acc > acc_max:
        acc_max = acc
        # save the model
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(network.state_dict(), model_path + network.name + '.pth')
    elif acc < acc_max - stop_delta:
        stop_count += 1
        print('\tno imporvement. count:', stop_count)


def save_record():
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    dic = {'train_losses': train_losses,
           'train_counter': train_counter,
           'valid_losses': valid_losses,
           'valid_acc': valid_acc,
           'valid_counter': valid_counter,
           'epochs': n_epochs}
    torch.save(dic, model_path + network.name + '_record.csv')


def load_record():
    global network, optimizer, scheduler, train_losses, train_counter, valid_losses, valid_acc, valid_counter
    network.load_state_dict(torch.load(model_path + network.name + '.pth'))
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=options.regularization)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: lr_lamb**epoch)
    dic = torch.load(model_path + network.name + '_record.csv')
    train_losses = dic['train_losses']
    train_counter = dic['train_counter']
    valid_losses = dic['valid_losses']
    valid_acc = dic['valid_acc']
    valid_counter = dic['valid_counter']
    return dic['epochs'] + 1


def plot_figure():
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(valid_counter, valid_losses, color='red')
    plt.scatter(valid_counter, valid_acc, color='green')
    plt.legend(['Train Loss', 'Test Loss', 'Test Accuracy'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('cross entropy loss')
    plt.savefig(network.name + ' ' + str(time.localtime().tm_mon)+'-'+str(time.localtime().tm_mday) + ' ' + str(time.localtime().tm_hour)+'-'+str(time.localtime().tm_min)+'-'+str(time.localtime().tm_sec)+'.png')
    plt.show()


if __name__ == "__main__":
    # train the model
    if options.train:
        start_epoch = 1
        # if continue -> load
        if options.continuing:
            start_epoch = load_record()

        # train
        start = time.time()
        for epoch in range(start_epoch, n_epochs + 1):
            print("-" * 40)
            train(epoch)
            test(epoch)
            if stop_count >= max_stop_count:
                print("\nNo improvement for {} epoches. Early stop!\n".format(max_stop_count))
                break
        print("-" * 40)
        print('\nFinished training! Total cost time: {}\n'.format(time.time()-start))

    # save record
    save_record()

    # plot a figure
    plot_figure()
