# Source: https://github.com/xdshang/real-time-object-detection/blob/master/utils.py

import cv2


def draw_result(img, result, scale=1.0):
    img_cp = img.copy()
    for i in range(len(result)):
        x_min = int(result[i][1]*scale)
        y_min = int(result[i][2]*scale)
        x_max = int(result[i][3]*scale)
        y_max = int(result[i][4]*scale)

        cv2.rectangle(img_cp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.rectangle(img_cp, (x_min, y_min-20), (x_max, y_min), (125, 125, 125), -1)
        cv2.putText(img_cp, result[i][0] + ' : %.2f' % result[i][5], (x_min+5, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_cp


def yield_frames_from_video(vs, mirror=False):
    while True:
        img = vs.read()
        if img is None:
            break
        if mirror:
            img = cv2.flip(img, 1)
        # crop image to square as YOLO input
        if img.shape[0] < img.shape[1]:
            pad = (img.shape[1]-img.shape[0])//2
            img = img[:, pad: pad+img.shape[0]]
        else:
            pad = (img.shape[0]-img.shape[1])//2
            img = img[pad: pad+img.shape[1], :]
        yield img
