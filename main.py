import time

import numpy as np
import cv2

maxSize = 700
xPoints = 450
yPoints = 800


def addNoise(image):
    height, width, colors = image.shape

    randNoize = np.random.randn(height, width, colors)
    randNoize = randNoize.reshape(height, width, colors)

    noisilyImage = image + image * randNoize * 1.2

    return noisilyImage


def createNoize(path, outPutPath):
    image = cv2.imread(path, 3)

    resultImage = addNoise(image)

    cv2.imwrite(outPutPath, resultImage)


def imageTracker(videoPath, imagePath):
    video = cv2.VideoCapture(videoPath)
    image = cv2.imread(imagePath, 3)

    downPoints = (xPoints, yPoints)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # output = cv2.VideoWriter("output.avi", fourcc, 20.0, downPoints)

    detector = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=100)

    i = 0
    while True:
        startTime = time.perf_counter()

        ret, frame = video.read(cv2.IMREAD_GRAYSCALE)
        if not ret:
            break

        frame = cv2.resize(frame, downPoints, interpolation=cv2.INTER_LINEAR)

        mask = detector.apply(frame)

        thresh = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 10)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if boolean2:
            for cont in contours:
                size = cv2.contourArea(cont)
                if size > maxSize:
                    x, y, w, h = cv2.boundingRect(cont)
                    if (x + w/2) < xPoints/2 and (y + h/2) < yPoints/2:
                        color = (255, 0, 0)
                    elif (x + w/2) > xPoints/2 and (y + h/2) > yPoints/2:
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.drawContours(frame, [cont], -1, color, 2)
        else:
            if len(contours) != 0:
                cont = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cont)
                if (x + w / 2) < xPoints / 2 and (y + h / 2) < yPoints / 2:
                    color = (255, 0, 0)
                elif (x + w / 2) > xPoints / 2 and (y + h / 2) > yPoints / 2:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.drawContours(frame, [cont], -1, color, 2)

        if boolean:
            cv2.imshow('frame', frame)
        else:
            cv2.imshow('frame', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        renderTime = time.perf_counter() - startTime
        # print("Render took " + str(renderTime))

        frameTime = 1/30 - renderTime
        if frameTime < 0:
            print("Render time too long - " + str(renderTime) + ". " + str(-frameTime))
        else:
            time.sleep(frameTime)

    video.release()


if __name__ == "__main__":
    inp = input()
    inp2 = input()
    if inp == 'f':
        boolean = True
    else:
        boolean = False
    if inp2 == 'y':
        boolean2 = True
    else:
        boolean2 = False

    # createNoize(path='image.jpg', outPutPath="newImage.jpg")

    imageTracker("video.MP4", "newImage.jpg")

cv2.destroyAllWindows()
