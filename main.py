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
parser.add_argument('--model', type=str, default='CNN', help='Model Name')
parser.add_argument('--train', default=False, action='store_true', help='Train or not')
parser.add_argument('--continue', default=False, action='store_true', help='Continue on the last training model or not', dest='continuing')
parser.add_argument('--report', default=False, action='store_true', help='Report all models\' score')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
parser.add_argument('--patience', type=int, default=7, help='How long to wait after last time validation loss improved')
options = parser.parse_args()
print(options)

train_file = './Dataset/train.txt'
valid_file = './Dataset/validation.txt'
test_file = './Dataset/test.txt'
word2vec_file = './Dataset/wiki_word2vec_50.bin'
model_path = './Model/'
img_path = './img/'
num_workers=0
n_epochs = options.epochs
batch_size_trian = 256
batch_size_valid = 512
batch_size_test = 512
random_seed = 2023
record_size = 1024
max_stop_count = options.patience
step_size = 4
max_lr, base_lr = {'MLP': (0.0002, 0.00004), 'CNN': (0.008, 0.0016), 'RNN': (0.0013, 0.00026)}[options.model]
learning_rate = (max_lr + base_lr) / 2


torch.manual_seed(random_seed)


# data loader
print('\nLoading data...')
train_set = My_Dataset(train_file, word2vec_file)
valid_set = My_Dataset(valid_file, word2vec_file)
test_set = My_Dataset(test_file, word2vec_file)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size_trian, shuffle=True, num_workers=num_workers, pin_memory=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size_valid, shuffle=True, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=True, num_workers=num_workers, pin_memory=True)
print('Data loaded!\n')


# build model
network = getattr(models, options.model)()
if torch.cuda.is_available():
    network.cuda()
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, step_size_down=step_size, cycle_momentum=False)
criterion = nn.CrossEntropyLoss()


# training process
train_losses = []
train_counter = []
valid_losses = []
valid_acc = []
valid_counter = []
loss_min = None
stop_count = 0

def train(epoch):
    network.train()
    record_loss = 0
    record_batch = 0
    print('Train Epoch', epoch)
    print("learning: {:.8f}".format(optimizer.param_groups[0]['lr']))
    for batch_idx, (data, target, input_len) in enumerate(train_loader):
        # calculate
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = network(data, input_len)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        #record
        record_loss += loss.item()
        record_batch += 1
        if batch_idx*batch_size_trian % record_size == 0:
            record_loss /= record_batch
            train_losses.append(record_loss)
            train_counter.append((batch_idx * batch_size_trian) + ((epoch - 1) * len(train_loader.dataset)))
            record_loss = 0
            record_batch = 0
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
    print('valid loss: {:.4f}'.format(loss))
    print('valid accuracy: {}/{} ({:.2f}%)'.format(correct, total, 100.*correct/total))

    # record
    valid_losses.append(loss)
    valid_acc.append(correct / total)
    valid_counter.append(epoch * len(train_loader.dataset))
    
    # early stop
    if loss_min == None or loss <= loss_min:
        loss_min = loss
        stop_count = 0
        # save the model
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(network.state_dict(), model_path + network.name + '.pth')
    else:
        stop_count += 1
        print(f'\tEarlyStopping counter: {stop_count} out of {max_stop_count}')


def report(data_loader):
    network.eval()
    TP = TN = FN = FP = 0
    with torch.no_grad():
        for data, target, input_len in data_loader:
            # calculate
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = network(data, input_len)
            pred = output.detach().max(1)[1]
            TP += ((pred == 1) & (target == 1)).sum().item()
            TN += ((pred == 0) & (target == 0)).sum().item()
            FN += ((pred == 0) & (target == 1)).sum().item()
            FP += ((pred == 1) & (target == 0)).sum().item()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    acc = (TP + TN) / (TP + TN + FP + FN)
    fscore = 2 / (1 / precision + 1 / recall)
    print(network.name + ' score:')
    print("Accuracy:\t{:.4f}".format(acc))
    print("F-score:\t{:.4f}\n".format(fscore))


def report_all():
    global network
    model_names = ['MLP', 'CNN', 'RNN']
    for name in model_names:
        network = getattr(models, name)()
        if torch.cuda.is_available():
            network.cuda()
        network.load_state_dict(torch.load(model_path + network.name + '.pth'))
        report(test_loader)


def save_record(last_epoch):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    dic = {'train_losses': train_losses,
           'train_counter': train_counter,
           'valid_losses': valid_losses,
           'valid_acc': valid_acc,
           'valid_counter': valid_counter,
           'epochs': last_epoch}
    torch.save(dic, model_path + network.name + '_record.csv')


def load_record():
    global network, optimizer, scheduler, train_losses, train_counter, valid_losses, valid_acc, valid_counter
    network.load_state_dict(torch.load(model_path + network.name + '.pth'))
    optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, step_size_down=step_size, cycle_momentum=False)
    dic = torch.load(model_path + network.name + '_record.csv')
    train_losses = dic['train_losses']
    train_counter = dic['train_counter']
    valid_losses = dic['valid_losses']
    valid_acc = dic['valid_acc']
    valid_counter = dic['valid_counter']
    return dic['epochs'] + 1


def plot_figure():
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(valid_counter, valid_losses, color='red')
    plt.plot(valid_counter, valid_acc, color='green')
    plt.legend(['Train Loss', 'Valid Loss', 'Valid Accuracy'], loc='right')
    plt.title(network.name)
    plt.xlabel('number of training examples seen')
    plt.ylabel('cross entropy loss')
    plt.savefig(img_path + network.name + ' ' + str(len(os.listdir(img_path))) +'.png')
    # plt.show()


if __name__ == "__main__":
    # train the model
    epoch = 0
    if options.train:
        start_epoch = 1
        # if continue -> load
        if options.continuing:
            start_epoch = load_record()

        # train
        for epoch in range(start_epoch, n_epochs + 1):
            print("-" * 40)
            train(epoch)
            test(epoch)
            if stop_count >= max_stop_count:
                print("\nNo improvement for {} epoches. Early stop!\n".format(max_stop_count))
                break
        print("-" * 40)
        report(test_loader)

    # report
    if options.report:
        report_all()

    if options.train:
        # save record
        save_record(epoch)

        # plot a figure
        plot_figure()
