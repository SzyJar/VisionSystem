import tkinter as tk
import cv2 as cv
import numpy as np


def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv.drawContours(drawing, contours, i, color, 1, cv.LINE_8, hierarchy, 0)
    # Show in a window
    cv.imshow('Contours', drawing)

webcam = cv.VideoCapture(0)

while True:
    ok,frame = webcam.read()

    if ok == True:
        cv.imshow("Source", frame)

        src_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))

        max_thresh = 255
        thresh = 225 # initial threshold
        thresh_callback(thresh)

        key = cv.waitKey(1)
        if key == ord("q"):
            break
webcam.release()
cv.destroyAllWindows