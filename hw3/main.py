import time
import numpy as np
import cv2

# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    u = np.concatenate([u, np.ones((N, 1))], axis=1)
    # Construct MatrixA
    for row in range(N):
        A[2*row, 0:3] = u[row]
        A[2*row+1, 3:6] = u[row]
        A[2*row:2*row+2, 6:9] = -np.matmul(v[row, np.newaxis].T, u[row, np.newaxis])
    U, _, _ = np.linalg.svd(np.matmul(A.T, A))
    H = U[:, -1].reshape((3, 3))
    return H

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    height, width, ch = img.shape
    img_corners = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
    projection_matrix = solve_homography(img_corners, corners)
    x_coor, y_coor = np.meshgrid(np.arange(width), np.arange(height).T)
    pad = np.ones(x_coor.shape)
    coor = np.stack((x_coor, y_coor, pad), axis=2)
    coor = coor.reshape((-1, 3))
    new_coor = np.matmul(projection_matrix, coor.T)
    new_coor = np.stack([np.divide(new_coor[0, :], new_coor[2, :]), \
                         np.divide(new_coor[1, :], new_coor[2, :])], axis=1)
    new_coor = new_coor.astype(int)
    canvas[new_coor[:, 1], new_coor[:, 0]] = img[y_coor.reshape(-1), x_coor.reshape(-1)]
    return canvas

def back_wrap(img, corners, part):
    if part == 'part2':
        height, width = 400, 400
    elif part == 'part3':
        height, width = 300, 500
    else:
        return None

    ret = np.zeros((height, width, 3))
    ret_corners = np.array([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]])
    projection_matrix = solve_homography(ret_corners, corners)
    x_coor, y_coor = np.meshgrid(np.arange(width), np.arange(height).T)
    pad = np.ones(x_coor.shape)
    coor = np.stack((x_coor, y_coor, pad), axis=2)
    coor = coor.reshape((-1, 3))
    new_coor = np.matmul(projection_matrix, coor.T)
    new_coor = np.stack([np.divide(new_coor[0, :], new_coor[2, :]), \
                         np.divide(new_coor[1, :], new_coor[2, :])], axis=1)
    new_coor = new_coor.astype(int)
    ret[y_coor.reshape(-1), x_coor.reshape(-1)] = img[new_coor[:, 0], new_coor[:, 1]]


    return ret

def main():
    # Part 1
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpeg')
    img2 = cv2.imread('./input/kuo.jpg')
    img3 = cv2.imread('./input/haung.jpg')
    img4 = cv2.imread('./input/tsai.jpg')
    img5 = cv2.imread('./input/han.jpg')

    canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])

    canvas = transform(img1, canvas, canvas_corners1)
    canvas = transform(img2, canvas, canvas_corners2)
    canvas = transform(img3, canvas, canvas_corners3)
    canvas = transform(img4, canvas, canvas_corners4)
    canvas = transform(img5, canvas, canvas_corners5)

    cv2.imwrite('part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 2
    ts = time.time()
    img = cv2.imread('./input/QR_code.jpg')
    corners  = np.array([[1237, 1980], [1395, 2025], [1212, 2040], [1365, 2085]])
    output2 = back_wrap(img, corners, part='part2')

    cv2.imwrite('part2.png', output2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

    # Part 3
    ts = time.time()
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    corners  = np.array([[142, 135], [138, 590], [320, 0], [313, 724]])
    output3 = back_wrap(img_front, corners, part='part3')

    cv2.imwrite('part3.png', output3)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

if __name__ == '__main__':
    main()
