import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def read_data(path):
    data, label = [], []
    for i in range(1, 41):
        for j in range(1, 11):
            img = cv2.imread(os.path.join(path, '{}_{}.png'.format(i, j)), cv2.IMREAD_GRAYSCALE)
            data.append(img)
            label.append(i)
    return np.array(data), np.array(label)

def pca(data):
    mean = np.mean(data, axis=0)
    x = data - mean
    eigen_vec, _, _ = np.linalg.svd(x.T, full_matrices=False)
    return mean, eigen_vec.T

def gram_pca(data):
    # flatten
    data = data.reshape(data.shape[0], -1)
    mean = np.mean(data, axis=0)
    data = data - mean # data shape(n_data, 46*56)
    # Calculate the eigen value for the gram matrix
    eigen_val, eigen_vec = np.linalg.eigh(np.dot(data, data.T))
    eigen_vec = np.dot(data.T, eigen_vec).T # eigen_vec shape(n_data, 46*56))
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / np.sqrt(eigen_val)
    return mean , eigen_vec[::-1, :]

def main():
    data, label = read_data(sys.argv[1])
    train_id = []
    for i in range(40):
        for j in range(7):
            train_id.append(i*10+j)
    mean, vec = pca(data[train_id].reshape(len(train_id), -1))
    # uncomment the code to run gram trick pca
    # mean, vec = gram_pca(data[train_id])

    # plot mean face and top 5 eigen face
    fig = plt.figure()
    mean_face = fig.add_subplot(2, 3, 1)
    mean_face.set_title('mean face')
    mean_face.imshow(mean.reshape(56, 46), cmap='gray')
    mean_face.axes.get_xaxis().set_visible(False)
    mean_face.axes.get_yaxis().set_visible(False)
    for i in range(5):
        face = fig.add_subplot(2, 3, i+2)
        face.set_title('face {}'.format(i+1))
        face.imshow(vec[i].reshape(56, 46), cmap='gray')
        face.axes.get_xaxis().set_visible(False)
        face.axes.get_yaxis().set_visible(False)
    plt.savefig(sys.argv[2]+'/2-3-1.jpg')
    plt.close()

    # plot reconstruct person8_image6
    fig = plt.figure()
    plt.subplots_adjust(wspace=0, hspace=0.3)
    person = cv2.imread(os.path.join(sys.argv[1], '8_6.png'), cv2.IMREAD_GRAYSCALE)
    person = person.reshape(-1) - mean
    weights = np.dot(vec, person)
    eigen_dim = [5, 50, 150, 280]
    # for gram trick pca
    # eigen_dim = [5, 50, 150, 279]
    for i in range(4):
        reconstruct = np.dot(weights[:eigen_dim[i]], vec[:eigen_dim[i]])
        ax = fig.add_subplot(2, 2, i+1)
        ax.set_title('reconstruct with {} faces\nmse = {:.2f}'\
            .format(eigen_dim[i], np.mean(np.square(reconstruct - person))))
        reconstruct = reconstruct + mean
        ax.imshow(reconstruct.reshape(56, 46), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.savefig(sys.argv[2]+'/2-3-2.jpg')
    plt.close()

    # plot dim=100 scatter
    mark = ['o', '2', '^', '8', 'v', '1', 'p', 'x', 's', '3']
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
    test_id = []
    test_label = []
    np.random.seed(65)
    for i in range(40):
        for j in range(7, 10):
            test_id.append(i*10+j)
            test_label.append(i)
    test_data = data[test_id].reshape(-1, 120)
    test_label = np.array(test_label)
    test_data = np.dot(vec[:100], test_data).T
    reduced_coor = TSNE(n_components=2).fit_transform(test_data, test_label)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('reduce dimension with tSNE')
    for i in range(40):
        ax.scatter(x=reduced_coor[3*i:3*(i+1), 0],
                   y=reduced_coor[3*i:3*(i+1), 1],
                   c=color[i%7], marker=mark[i%10])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.savefig(sys.argv[2]+'/2-3-3.jpg')
    plt.close()

if __name__ == '__main__':
    main()
