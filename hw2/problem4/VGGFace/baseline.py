###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import os
import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torchvision.models import alexnet
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torch

if __name__ == "__main__":
    np.random.seed(65)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.ToTensor(), normalize])
    #dir_path = './hw2-4_data/problem2/'
    dir_path = sys.argv[1]
    batch_size = 1
    train_path = os.path.join(dir_path, 'train/')
    train_set = ImageFolder(train_path, transform=trans)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_path = os.path.join(dir_path, 'valid/')
    valid_set =  ImageFolder(valid_path,  transform=trans)
    valid_loader  = torch.utils.data.DataLoader(dataset=valid_set,  batch_size=batch_size, shuffle=False)

    device = torch.device('cuda')
    extractor = alexnet(pretrained=True).features
    extractor.to(device)
    extractor.eval()

    train_feature, train_label = [], []
    with torch.no_grad():
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            feat = extractor(img).view(img.size(0), 256, -1)
            feat = torch.mean(feat, 2)
            feat = feat.cpu().numpy()
            train_feature.append(feat)
            train_label.append(label.item())

    valid_feature, valid_label = [], []
    with torch.no_grad():
        for img, label in valid_loader:
            img, label = img.to(device), label.to(device)
            feat = extractor(img).view(img.size(0), 256, -1)
            feat = torch.mean(feat, 2)
            feat = feat.cpu().numpy()
            valid_feature.append(feat)
            valid_label.append(label.item())
    
    train_feature, valid_feature = np.array(train_feature).squeeze(1), np.array(valid_feature).squeeze(1)

    pca = PCA(n_components=100)
    train_feature = pca.fit_transform(X=train_feature)
    valid_feature = pca.transform(X=valid_feature)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X=train_feature, y=train_label)
    print(knn.score(X=valid_feature, y=valid_label))


    # Plot Scatter
    color = ['xkcd:purple', 'xkcd:green', 'xkcd:blue', 'xkcd:pink', 'xkcd:brown',\
             'xkcd:sky blue', 'xkcd:teal', 'xkcd:orange', 'xkcd:yellow', 'xkcd:grey']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('TSNE for AlexNet')

    reduced_coor = TSNE(n_components=2).fit_transform(X=train_feature, y=train_label)
    for i, identity in enumerate(train_label):
        if identity < 10:
            ax.scatter(x=reduced_coor[i, 0], y=reduced_coor[i, 1], c=color[identity])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

    reduced_coor = TSNE(n_components=2).fit_transform(X=valid_feature, y=valid_label)
    for i, identity in enumerate(valid_label):
        if identity < 10:
            ax.scatter(x=reduced_coor[i, 0], y=reduced_coor[i, 1], c=color[identity], marker='x')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    plt.savefig(sys.argv[2])
    plt.close()
