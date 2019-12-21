import numpy as np
import cv2
import sys

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

def main(ref_image,template,video):
    ref_image = cv2.imread(ref_image)
    template = cv2.imread(template)
    template = cv2.resize(template, ref_image.shape[:2] , interpolation=cv2.INTER_CUBIC)

    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))

    i = 0
    sift = cv2.xfeatures2d.SIFT_create()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}'.format(i))
        if ret:  
            ## check whethere the frame is legal, i.e., there still exists a frame
            ## Feature Matching
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(frame, None)
            matches = flann.knnMatch(des1, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance < 0.85 * n.distance:
                    good.append(m)

            ## Homography
            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                height, width = template.shape[:2]

                x_coor, y_coor = np.meshgrid(np.arange(width), np.arange(height).T)
                pad = np.ones(x_coor.shape)
                coor = np.stack((x_coor, y_coor, pad), axis=2)
                coor = coor.reshape((-1, 3))

                new_coor = np.matmul(H, coor.T)
                new_coor = np.stack([np.divide(new_coor[0, :], new_coor[2, :]), \
                                     np.divide(new_coor[1, :], new_coor[2, :])], axis=1)
                new_coor = new_coor.astype(int)
                new_coor[new_coor[:, 0] >= film_w, 0] = film_w - 1
                new_coor[new_coor[:, 0] <= 0, 0] = 0
                new_coor[new_coor[:, 1] >= film_h, 1] = film_h - 1
                new_coor[new_coor[:, 0] <= 0, 0] = 0

                frame[new_coor[:, 1], new_coor[:, 0]] \
                = ref_image[y_coor.reshape(-1), x_coor.reshape(-1)]
            videowriter.write(frame)
            i = i + 1

        else:
            break
            
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = sys.argv[1]  ## path to ar_marker.mp4
    main(ref_path,template_path,video_path)