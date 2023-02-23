import tensorflow as tf

import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import cv2 as cv


model = tf.keras.models.load_model('saved_model/97_percent')
model.summary()

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

    input_tensor = []
    # draw detected objects
    for obj in range(len(objects)):
        skip = 0
        x, y, w, h = objects[obj]
        for i in range(len(objects)):
            if x > objects[i][0] and (x + w) < (objects[i][0] + objects[i][2]) and y > objects[i][1] and (y + h) < (objects[i][1] + objects[i][3]):
                skip = 1
        if skip == 0:
            cv.rectangle(resultImg, (x, y), (x + w, y + h), (0, 255, 0), 1)
            detectedObjects.append(frame[(y):(y + h), (x):(x + w)])
            detectedObjects[-1] = cv.resize(detectedObjects[-1], (150,150))
            image_np = np.array(detectedObjects[-1])
            input_tensor.append(tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32))
            coords.append([x,y])     

    # identify detected objects  
    probability_model = tf.keras.Sequential([model, 
                                            tf.keras.layers.Softmax()])
    predictions = []
    for i in range(len(input_tensor)):
        predictions.append(probability_model.predict(input_tensor[i]))

    predictionPercent = []
    predictionLabel = []
    name = ["button", "clip", "dark glass", "frame", "light glass", "rectangle", "strip"]
    for i in range(len(predictions)):
        value = 0
        category = 0
        for j in range(len(predictions[i][0])):
            if predictions[i][0][j] > value:
                value = predictions[i][0][j]
                category = j
        predictionPercent.append(value)
        predictionLabel.append(name[category])
        cv.putText(img = resultImg, text = f"{predictionLabel[i]} {int(predictionPercent[i]*100)}%",
                            org = (coords[i][0], coords[i][1]), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                            fontScale = 0.5, color = (0, 255, 0), thickness = 1)

    # Show in a window
    cv.imshow("Source frame", frame)
    cv.imshow("Contour detection", drawing)
    cv.imshow("Object detection", resultImg)


# contour deteciotn parameters
def empty(a):
    pass
cv.namedWindow("Params")
cv.resizeWindow("Params", 640, 240)
cv.createTrackbar("thresh1", "Params", 80, 255, empty)
cv.createTrackbar("thresh2", "Params", 230, 500, empty)
cv.createTrackbar("AreaMin", "Params", 400, 30000, empty)
cv.createTrackbar("AreaMax", "Params", 12000, 30000, empty)

# start camera
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