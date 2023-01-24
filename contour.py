import tkinter as tk
import cv2 as cv
import numpy as np


def detection(val):
    threshold = val
    resultImg = (frame).copy()
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv.drawContours(drawing, contours, i, color, 1, cv.LINE_8, hierarchy, 0)
        # Object detection
        cnt = contours[i]
        (x,y),radius = cv.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        if(int(radius) > 1 and int(radius) < 80):
            #draw the rectangle
            startPoint = (int(center[0] - radius), int(center[1] - radius))
            endPoint = (int(center[0] + radius), int(center[1] + radius))
            resultImg = cv.rectangle(resultImg, startPoint, endPoint, (0, 255, 0), 1)
            cv.putText(img = resultImg, text = f"{center[0]}, {center[1]}", org = center, fontFace = cv.FONT_HERSHEY_TRIPLEX,
                        fontScale = 0.5, color = (0, 255, 0), thickness = 1)

    # Show in a window
    cv.imshow("Source frame", frame)
    cv.imshow("Contour detection", drawing)
    cv.imshow("Object detection", resultImg)


webcam = cv.VideoCapture(0)

while True:
    ok,frame = webcam.read()

    if ok == True:
        src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))

        thresh = 250 # initial threshold
        detection(thresh)

        key = cv.waitKey(1)
        if key == ord("q"):
            break

webcam.release()
cv.destroyAllWindows