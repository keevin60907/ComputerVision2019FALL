import numpy as np
import cv2

class Joint_bilateral_filter(object):

    def __init__(self, sigma_s, sigma_r, border_type='reflect'):

        self.border_type = border_type
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s

    def joint_bilateral_filter(self, input, guidance):

        r = int(3 * self.sigma_s)
        kernel = int(2 * r + 1)
        guidance = guidance / 255
        output = np.zeros(input.shape)

        # padding
        input = cv2.copyMakeBorder(input, r, r, r, r,cv2.BORDER_REFLECT)
        guidance = cv2.copyMakeBorder(guidance, r, r, r, r,cv2.BORDER_REFLECT)
        if len(guidance.shape) == 2:
            guidance = np.expand_dims(guidance, axis=2)

        # bulid G_s (independent to guidance)
        bound = np.arange(2 * r + 1) - r    # [-r .. 0 .. r]
        x_coor, y_coor = np.meshgrid(bound, -bound.T)
        G_s = np.square(x_coor) + np.square(y_coor)
        G_s = np.exp(-G_s / (2 * np.square(self.sigma_s)))

        # brute force conv
        for row in range(output.shape[0]):
            for col in range(output.shape[1]):
                # bulid G_r (depends on guidance)
                G_r = guidance[row:row+kernel, col:col+kernel, :] - guidance[row+r, col+r, :]
                G_r = np.sum(np.square(G_r), axis=2)
                G_r = np.exp(-G_r / (2 * np.square(self.sigma_r)))
                # make window on input image
                window = input[row:row+kernel, col:col+kernel, :]
                # convolution
                filter = np.multiply(G_s, G_r)
                denominator = np.sum(filter)
                output[row, col, 0] = np.sum(np.multiply(filter, window[:, :, 0])) / denominator
                output[row, col, 1] = np.sum(np.multiply(filter, window[:, :, 1])) / denominator
                output[row, col, 2] = np.sum(np.multiply(filter, window[:, :, 2])) / denominator

        return output
