import torch
import cv2
import sys
import os
import numpy as np
from util import resnet
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

EPOCH = 20
'''
python my_train.py hw2-4_data/problem2/
'''

def train():
    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []

    np.random.seed(65)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    #dir_path = './hw2-4_data/problem2/'
    dir_path = sys.argv[1]
    batch_size = 64
    train_path = os.path.join(dir_path, 'train/')
    train_set = ImageFolder(train_path, transform=trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_path = os.path.join(dir_path, 'valid/')
    valid_set =  ImageFolder(valid_path,  transform=trans)
    valid_loader  = torch.utils.data.DataLoader(dataset=valid_set,  batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')
    model = resnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCH):
        model.train()
        train_loss = 0
        correct = 0
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = F.cross_entropy(output, label)
            train_loss += F.cross_entropy(output, label).item()
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()
            if batch_idx % 10 == 0:
                print('Train Epoch: {:0>2d} [{:0>5d}/{:0>5d} ({:.0f}%)]     Loss: {:.6f}\r'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()), end='')
            if batch_idx == len(train_loader)-1:
                print()
                train_loss /= len(train_loader.dataset)
                print('Train set: Average loss: {:.4f}, Accuracy: {:0>4d}/{:0>4d} ({:.0f}%)'.format(
                    train_loss, correct, len(train_loader.dataset),
                    100. * correct / len(train_loader.dataset)))
                train_loss_history.append(train_loss)
                train_acc_history.append(correct / len(train_loader.dataset))

        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in valid_loader:
                data, label = data.to(device), label.to(device)
                output, _ = model(data)
                valid_loss += F.cross_entropy(output, label).item()
                pred = output.max(1, keepdim=True)[1]   
                correct += pred.eq(label.view_as(pred)).sum().item()

        valid_loss /= len(valid_loader.dataset)
        print('valid set: Average loss: {:.4f}, Accuracy: {:0>4d}/{:0>4d} ({:.0f}%)'.format(
            valid_loss, correct, len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))
        valid_loss_history.append(valid_loss)
        valid_acc_history.append(correct / len(valid_loader.dataset))

        torch.save(model.state_dict(), './model/model_{}.pkl'.format(epoch))

    plt.figure()
    plt.plot(range(EPOCH), train_loss_history, c='blue', label='train')
    plt.plot(range(EPOCH), valid_loss_history, c='red', label='valid')
    plt.title('Loss Curve')
    plt.xlabel('epoch')
    plt.ylabel('cross entropy')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

    # Plot Accuracy Curve
    plt.figure()
    plt.plot(range(EPOCH), train_acc_history, c='blue', label='train')
    plt.plot(range(EPOCH), valid_acc_history, c='red', label='valid')
    plt.title('Accuracy Curve')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('acc.png')
    plt.close()

if __name__ == '__main__':
    train()
