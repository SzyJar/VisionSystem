import cv2 as cv
import numpy as np


def detection(tresh1, tresh2):
    resultImg = (frame).copy()
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, tresh1, tresh2)
    kernel = np.ones((5,5))
    imgDil = cv.dilate(canny_output, kernel, iterations = 2)
    # Find contours
    contours, hierarchy = cv.findContours(imgDil, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # Draw contours
    drawing = np.zeros((imgDil.shape[0], imgDil.shape[1], 3), dtype=np.uint8)
    objects = []
    detectedObjects = []
    coords = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        areaMin = cv.getTrackbarPos("AreaMin", "Params")
        areaMax = cv.getTrackbarPos("AreaMax", "Params")
        if area > areaMin and area < areaMax:
            cv.drawContours(drawing, cnt, -1, (255, 255, 2550), 1)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv.boundingRect(approx)
            # add object to list
            objects.append([int(x), int(y), int(w), int(h)])

    # draw detected objects
    for obj in range(len(objects)):
        skip = 0
        x, y, w, h = objects[obj]
        for i in range(len(objects)):
            if x > objects[i][0] and (x + w) < (objects[i][0] + objects[i][2]) and y > objects[i][1] and (y + h) < (objects[i][1] + objects[i][3]):
                skip = 1
        if skip == 0:
            cv.rectangle(resultImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
            detectedObjects.append(frame[y:(y + h), x:(x + w)])
            coords.append([x,y])

    # identify detected objects
    for obj in range(len(detectedObjects)):
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_button,None)
        kp2, des2 = sift.detectAndCompute(detectedObjects[obj],None)
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        if len(good) > 10:
            cv.putText(img = resultImg, text = "button",
                        org = (coords[obj][0], coords[obj][1]), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                        fontScale = 1, color = (0, 255, 0), thickness = 1)
    """
    for i in range(len(contours)):
        color = (255, 255, 255)
        cv.drawContours(drawing, contours, i, color, 1, cv.LINE_8, hierarchy, 0)        
        # Object detection
        cnt = contours[i]
        (x,y),radius = cv.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        if(int(radius) > 1 and int(radius) < 80):
            # Draw rectangle
            startPoint = (int(center[0] - radius), int(center[1] - radius))
            endPoint = (int(center[0] + radius)), int(center[1] + radius)
            resultImg = cv.rectangle(resultImg, startPoint, endPoint, (0, 255, 0), 1)
            cv.putText(img = resultImg, text = "",#f"{center[0]}, {center[1]}",
                         org = (center[0], center[1]), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                        fontScale = 1, color = (0, 255, 0), thickness = 1)
        """

    # Show in a window
    cv.imshow("Source frame", frame)
    cv.imshow("Contour detection", drawing)
    cv.imshow("Object detection", resultImg)
    try:
        cv.imshow("object", detectedObjects[cv.getTrackbarPos("item", "Params")])
    except:
        cv.destroyWindow("object")


def empty(a):
    pass

cv.namedWindow("Params")
cv.resizeWindow("Params", 640, 240)
cv.createTrackbar("thresh1", "Params", 190, 255, empty)
cv.createTrackbar("thresh2", "Params", 110, 500, empty)
cv.createTrackbar("AreaMin", "Params", 400, 30000, empty)
cv.createTrackbar("AreaMax", "Params", 12000, 30000, empty)
cv.createTrackbar("item", "Params", 1, 10, empty)

# load  train images

img_button = cv.imread('przycisk.jpg',cv.IMREAD_GRAYSCALE)
img_darkGlass = cv.imread('ciemneSzklo.jpg',cv.IMREAD_GRAYSCALE)
img_frame = cv.imread('obudowa.jpg',cv.IMREAD_GRAYSCALE)
img_strip = cv.imread('pasek.jpg',cv.IMREAD_GRAYSCALE)

webcam = cv.VideoCapture(0)

while True:
    button = False
    ok,frame = webcam.read()

    if ok == True:
        frameBlur = cv.GaussianBlur(frame, (7,7), 1)
        src_gray = cv.cvtColor(frameBlur, cv.COLOR_BGR2GRAY)
        #src_gray = cv.blur(src_gray, (3,3))

        tresh1 = cv.getTrackbarPos("thresh1", "Params")
        tresh2 = cv.getTrackbarPos("thresh2", "Params")
        
        detection(tresh1, tresh2)
        while button == False:
            key2 = cv.waitKey(2)
            if key2 == ord("w"):
                break

        key = cv.waitKey(1)
        if key == ord("q"):
            break


webcam.release()
cv.destroyAllWindows