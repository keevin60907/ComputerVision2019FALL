import numpy as np
import cv2
from joint_bilateral_filter import Joint_bilateral_filter

def lower_than_neighbor(w_r, w_g, w_b, costmap, key):
    if w_r > 10 or w_g > 10 or w_b > 10:
        return True
    elif w_r < 0 or w_g < 0 or w_b < 0:
        return True
    return costmap[key] < costmap['{},{},{}'.format(w_r, w_g, w_b)]

def check_min(costmap, key):
    w_r = int(key.split(',')[0])
    w_g = int(key.split(',')[1])
    w_b = int(key.split(',')[2])
    return lower_than_neighbor(w_r-1, w_g+1, w_b, costmap, key)\
           and lower_than_neighbor(w_r-1, w_g, w_b+1, costmap, key)\
           and lower_than_neighbor(w_r+1, w_g-1, w_b, costmap, key)\
           and lower_than_neighbor(w_r, w_g-1, w_b+1, costmap, key)\
           and lower_than_neighbor(w_r+1, w_g, w_b-1, costmap, key)\
           and lower_than_neighbor(w_r, w_g+1, w_b-1, costmap, key)

def cost_fn(img1, img2):
    return np.sum(np.abs(img1-img2))

def rgb2gray(img, w_r, w_g, w_b):
    # traditional rgb2gray (w_r, w_g, w_b) = (0.299, 0.587, 0.114)
    # img_gray = rgb2gray(img, 0.299, 0.587, 0.114)
    weights = np.ones(img.shape)
    weights[:, :, 0] = weights[:, :, 0] * w_r
    weights[:, :, 1] = weights[:, :, 1] * w_g
    weights[:, :, 2] = weights[:, :, 2] * w_b
    return np.sum(np.multiply(img, weights), axis=2)

def voting(img, sigma_s, sigma_r, vote):
    costmap = {} # make costmap of 66 points
    cnt = 0
    JBF = Joint_bilateral_filter(sigma_s, sigma_r, border_type='reflect')
    for w_r in range(11):
        for w_g in range(11 - w_r):
            print('Process {} points\r'.format(cnt), end='')
            w_b = 10 - w_r - w_g
            img_gray = rgb2gray(img, w_r/10., w_g/10., w_b/10.)
            img_recon = JBF.joint_bilateral_filter(img, img).astype(np.uint8)
            img_out = JBF.joint_bilateral_filter(img, img_gray).astype(np.uint8)
            costmap['{},{},{}'.format(w_r, w_g, w_b)] = cost_fn(img_recon, img_out)
            cnt = cnt + 1
    print()
    for key, value in costmap.items():
        if check_min(costmap, key):
            vote.append(key)

def main():
    log = open('vote.log', 'w')
    spatial_kernel = [1, 2, 3]
    range_kernel = [0.05, 0.1, 0.2]
    imgs = ['./testdata/2a.png', './testdata/2b.png', './testdata/2c.png']
    vote = []
    
    img = cv2.imread(imgs[1])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for sigma_s in spatial_kernel:
        for sigma_r in range_kernel:
            print('Sigma_s = {}, Sigma_r = {}'.format(sigma_s, sigma_r))
            voting(img_rgb, sigma_s, sigma_r, vote)
    print(vote, file=log)

    dict = {}
    for item in vote:
        if item in dict:
            dict[item] += 1
        else:
            dict[item] = 1
    print(dict, file=log)

if __name__ == '__main__':
    main()
