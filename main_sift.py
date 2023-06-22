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
            # Add object to list
            objects.append([int(x), int(y), int(w), int(h)])

    # Draw detected objects
    for obj in range(len(objects)):
        skip = 0
        x, y, w, h = objects[obj]
        for i in range(len(objects)):
            if x > objects[i][0] and (x + w) < (objects[i][0] + objects[i][2]) and y > objects[i][1] and (y + h) < (objects[i][1] + objects[i][3]):
                skip = 1
        if skip == 0:
            cv.rectangle(resultImg, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv.rectangle(resultImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
            detectedObjects.append(frame[(y):(y + h), (x):(x + w)])
            coords.append([x,y])

    # Identify detected objects
    sift = cv.SIFT_create() # Initiate SIFT detector
    for obj in range(len(detectedObjects)):
        detected = [] # Empty list for detected object information
        text = ''
        for test in range(len(img_test)):   
            # Find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(detectedObjects[obj],None)
            kp2, des2 = sift.detectAndCompute(img_test[test],None)

            bf = cv.BFMatcher()
    
            try:
                matches = bf.knnMatch(des1,des2,k=2)
            except:
                pass

            good = []
            try:
                for m,n in matches:
                    if m.distance < 0.7*n.distance:
                        good.append([m])
            except:
                print("not enough values to unpack (expected 2)")
                
            detected.append([testImageName[test], len(good)])

        j = 0
        for i in range(len(detected)):
            if(detected[i][1] > j and detected[i][1] >= cv.getTrackbarPos("matchCount", "Params")):
                j = detected[i][1]
                text = f"{detected[i][0]} matches = {detected[i][1]}"

        cv.putText(img = resultImg, text = text,
                    org = (coords[obj][0], coords[obj][1]), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                    fontScale = 0.5, color = (0, 255, 0), thickness = 1)
                

    # Show in a window
    cv.imshow("Source frame", frame)
    cv.imshow("Contour detection", drawing)
    cv.imshow("Object detection", resultImg)

# Contour deteciotn parameters
def empty(a):
    pass
cv.namedWindow("Params")
cv.resizeWindow("Params", 640, 240)
cv.createTrackbar("thresh1", "Params", 190, 255, empty)
cv.createTrackbar("thresh2", "Params", 210, 500, empty)
cv.createTrackbar("AreaMin", "Params", 400, 30000, empty)
cv.createTrackbar("AreaMax", "Params", 12000, 30000, empty)
cv.createTrackbar("matchCount", "Params", 10, 100, empty)

# Load  train images

img_darkGlass = cv.imread('./Images/pattern/darkGlass.jpg',cv.IMREAD_GRAYSCALE)
img_frame = cv.imread('./Images/pattern/frame.jpg',cv.IMREAD_GRAYSCALE)
img_strip = cv.imread('./Images/pattern/strip.jpg',cv.IMREAD_GRAYSCALE)
img_clip = cv.imread('./Images/pattern/clip.jpg',cv.IMREAD_GRAYSCALE)

img_test = [img_darkGlass, img_frame, img_strip, img_clip]
testImageName = ["dark glass", "frame", "strip", "clip"]

# Start camera
webcam = cv.VideoCapture(0)

while True:
    button = False
    ok,frame = webcam.read()

    if ok == True:
        frameBlur = cv.GaussianBlur(frame, (7,7), 1)
        src_gray = cv.cvtColor(frameBlur, cv.COLOR_BGR2GRAY)

        tresh1 = cv.getTrackbarPos("thresh1", "Params")
        tresh2 = cv.getTrackbarPos("thresh2", "Params")
        
        detection(tresh1, tresh2)
        key = cv.waitKey(1)

        if key == ord("q"):
            break

webcam.release()
cv.destroyAllWindows