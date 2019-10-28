import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def read_data(path):
    data, label = [], []
    for file in sorted(os.listdir(path)):
        if file[-4:] == '.png':
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            data.append(img)
            label.append(int(file.split('_')[0]))
    return np.array(data), np.array(label)

def PCA(data):
    mean = np.mean(data, axis=0)
    x = data - mean
    eigen_vec, _, _ = np.linalg.svd(x.T, full_matrices=False)
    return mean, eigen_vec.T

def main():
    dim = [4, 51, 101]
    data, label = read_data(sys.argv[1])

    train_id = []
    train_label = []
    for i in range(40):
        for j in range(5):
            train_id.append(i*10+j)
            train_label.append(i)
    valid_id = []
    valid_label = []
    for i in range(40):
        for j in range(5, 7):
            valid_id.append(i*10+j)
            valid_label.append(i)
    test_id = []
    test_label = []
    for i in range(40):
        for j in range(7, 10):
            test_id.append(i*10+j)
            test_label.append(i)

    mean, vec = PCA(data[train_id].reshape(len(train_id), -1))
    for k in [1, 3, 5]:
        for n in [3, 50, 100]:
            clf = KNeighborsClassifier(n_neighbors=k)
            train_data = data[train_id].reshape((len(train_id), -1))-mean
            train_data = np.dot(train_data, vec[:n].T)
            valid_data = data[valid_id].reshape((len(valid_id), -1))-mean
            valid_data = np.dot(valid_data, vec[:n].T)
            clf.fit(train_data, train_label)
            print('k = {}, n = {}, validation score = {:.4f}'\
                  .format(k, n, clf.score(valid_data, valid_label)))

    # Choose k = 1, n = 100
    train_id = []
    train_label = []
    for i in range(40):
        for j in range(7):
            train_id.append(i*10+j)
            train_label.append(i)
    mean, vec = PCA(data[train_id].reshape(len(train_id), -1))
    clf = KNeighborsClassifier(n_neighbors=1)
    train_data = data[train_id].reshape((len(train_id), -1))-mean
    train_data = np.dot(train_data, vec[:100].T)
    clf.fit(train_data, train_label)
    test_data = data[test_id].reshape((len(test_id), -1))-mean
    test_data = np.dot(test_data, vec[:100].T)
    print('test score = {:.4f}'.format(clf.score(test_data, test_label)))

if __name__ == '__main__':
    main()
