import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class KMeans_Classifier():

    def __init__(self, data, eigen_vec, k=1, iteration=1000):
        self.num_data = data.shape[0]
        self.num_center = k
        self.num_iter = iteration
        self.eigen_vec = eigen_vec
        self.data = data.reshape(data.shape[0], -1)
        self.data = self.data - np.mean(self.data, axis=0)
        self.eigen_data = np.dot(self.data, self.eigen_vec.T)
        self.centeroid = self.eigen_data[np.random.randint(self.num_data, size=k)]
        self.weight = np.array([0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    def euclidean(self, x, y):
        return np.sqrt(np.dot(np.square(x - y), self.weight.T))

    def cluster(self):
        self.group = [[] for _ in range(self.num_center)]
        mid_dis = np.inf
        for i in range(self.num_data):
            for j in range(self.num_center):
                distant = self.euclidean(self.eigen_data[i], self.centeroid[j])
                if distant < mid_dis:
                    mid_dis = distant
                    flag = j
            self.group[flag].append(i)
            mid_dis = np.inf

    def re_center(self):
        new_centeroid = []
        for index, objects in enumerate(self.group):
            if len(objects) == 0:
                new_centeroid.append(self.centeroid[index])
            else:
                new_centeroid.append(np.mean(self.eigen_data[objects], axis=0))
        new_centeroid = np.array(new_centeroid)
        ret = np.mean(np.square(new_centeroid - self.centeroid))
        self.centeroid = new_centeroid
        return ret

    def plot(self, num_iter):
        mark = ['o', '2', '^', '8', 'v', '1', 'p', 'x', 's', '3']
        color = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('iteration {:0>2d}'.format(num_iter))
        for index, objects in enumerate(self.group):
            ax.scatter(x=self.eigen_data[objects][:, 0],
                       y=self.eigen_data[objects][:, 1],
                       c=[color[index%7] for _ in objects],
                       marker=mark[index%10],
                       s=[5 for _ in objects], alpha=0.5)
            ax.scatter(x=self.centeroid[index][0], y=self.centeroid[index][1],
                       c=color[index%7], marker=mark[index%10])
        plt.savefig(sys.argv[2]+'/{:0>2d}_iteration.jpg'.format(num_iter))
        plt.close()

    def fit(self):
        for iter in range(self.num_iter):
            self.cluster()
            change = self.re_center()
            print('The {:0>4d} iteration, centeroid moves: {:.3f}'.format(iter, change))
            self.plot(iter+1)
            if change < 10e-6:
                break
        print('{}'.format('='*20))
        for idx, number in enumerate(self.group):
            print('group: {:0>2d}'.format(idx))
            people = ['{}_{}'.format((pic//7)+1, (pic%7)+1) for pic in number]
            for i in people:
                print('{}, '.format(i), end='')
            print('')
            print('{}'.format('='*20))

def read_data(path):
    data, label = [], []
    for i in range(1, 41):
        for j in range(1, 11):
            img = cv2.imread(os.path.join(path, '{}_{}.png'.format(i, j)), cv2.IMREAD_GRAYSCALE)
            data.append(img)
            label.append(i)
    return np.array(data), np.array(label)

def PCA(data):
    mean = np.mean(data, axis=0)
    x = data - mean
    eigen_vec, _, _ = np.linalg.svd(x.T, full_matrices=False)
    return mean, eigen_vec.T

def main():
    np.random.seed(37)
    data, label = read_data(sys.argv[1])
    train_id = []
    for i in range(10):
        for j in range(7):
            train_id.append(i*10+j)
    mean, vec = PCA(data[train_id].reshape(len(train_id), -1))
    clf = KMeans_Classifier(data[train_id], vec[:10], k=10)
    clf.fit()

if __name__ == '__main__':
    main()
