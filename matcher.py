import time

import numpy as np
import cv2

xPoints = 450
yPoints = 800
color = (0, 255, 0)
MIN_MATCH_COUNT = 1


def imageTracker(videoPath, imagePath):
    video = cv2.VideoCapture(videoPath)
    marker = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    downPoints = (xPoints, yPoints)

    sift = cv2.SIFT_create()
    kpMarker, desMarker = sift.detectAndCompute(marker, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    i = 0
    while True:
        boolean = False
        startTime = time.perf_counter()

        ret, frame = video.read(cv2.IMREAD_GRAYSCALE)
        if not ret:
            break

        frame = cv2.resize(frame, downPoints, interpolation=cv2.INTER_LINEAR)

        kp, des = sift.detectAndCompute(frame, None)

        matches = flann.knnMatch(des, desMarker, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            print(good)
            boolean = True

            src_pts = np.float32([kp[m.queryIdx].pt for m in good])
            print(src_pts)

            for m in src_pts:
                print(m)
                x = round(m[0])
                y = round(m[1])
                cv2.rectangle(frame, (x, y), (x + 5, y + 5), color, 2)

        renderTime = time.perf_counter() - startTime
        # print("Render took " + str(renderTime))

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if boolean:
        #     time.sleep(0.1)

        frameTime = 1/30 - renderTime
        if frameTime < 0:
            print("Render time too long - " + str(renderTime) + ". " + str(-frameTime))
        else:
            time.sleep(frameTime)

    video.release()


if __name__ == "__main__":
    imageTracker("video.MP4", "ref-point.jpg")