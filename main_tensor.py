import tensorflow as tf
from tensorflow.keras.preprocessing import image

import numpy as np
import cv2 as cv


model_path = "saved_model/test.h5"
model = tf.keras.models.load_model(model_path)
items = ["clip", "dark glass", "frame", "strip"]
desiredCoords = [[250,250], [400,200], [50,150], [470,400]]
model.summary()

class VisionSystem():
    def __init__(self):
        self.itemCat = []
        self.itemCoord = []

    def detection(self, tresh1, tresh2):
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
            areaMin = cv.getTrackbarPos("AreaMin", windowName)
            areaMax = cv.getTrackbarPos("AreaMax", windowName)
            if area > areaMin and area < areaMax:
                cv.drawContours(drawing, cnt, -1, (255, 255, 2550), 1)
                peri = cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
                x, y, w, h = cv.boundingRect(approx)
                # add object to list
                objects.append([int(x), int(y), int(w), int(h)])

        # get coords of detected objects
        input_image = []
        for obj in range(len(objects)):
            skip = 0
            x, y, w, h = objects[obj]
            for i in range(len(objects)):
                if x > objects[i][0] and (x + w) < (objects[i][0] + objects[i][2]) and y > objects[i][1] and (y + h) < (objects[i][1] + objects[i][3]):
                    skip = 1
            if skip == 0:
                detectedObjects.append(frame[(y):(y + h), (x):(x + w)])
                detectedObjects[-1] = cv.resize(detectedObjects[-1], (150,150))
                input_image.append(image.img_to_array(detectedObjects[-1]))
                input_image[-1] = np.expand_dims(input_image[-1], axis = 0)
                coords.append([x,y,w,h])

        """
        #show one of detected objects
        if(len(detectedObjects)>0):
            cv.imshow("Contour 231", detectedObjects[-1])    
        """

        # text color and position margin
        goodColor = (0, 255, 0)
        badColor = (50, 50, 255)
        margin = 20


        self.itemCat = []
        self.itemCoord = []
        # identify detected objects
        probability_model = tf.keras.Sequential([model, 
                                                tf.keras.layers.Softmax()])                          
        predictions = []
        for i in range(len(input_image)):
            predictions.append(probability_model.predict(input_image[i], verbose = 0))
        predictionPercent = []
        predictionLabel = []
        name = items
        for i in range(len(predictions)):
            value = 0
            category = 0
            for j in range(len(predictions[i][0])):
                if predictions[i][0][j] > value:
                    value = predictions[i][0][j]
                    category = j
            predictionPercent.append(value) 
            predictionLabel.append(name[category])

            # push prediction and coords outside this method
            self.itemCat.append(category) 
            self.itemCoord.append(coords[i])

            # check if position is correct 
            if(coords[i][0] > desiredCoords[category][0] - margin and coords[i][1] > desiredCoords[category][1] - margin and
                coords[i][0] < desiredCoords[category][0] + margin and coords[i][1] < desiredCoords[category][1] + margin):
                textColor = goodColor
            else:
                textColor = badColor
            cv.rectangle(resultImg, (coords[i][0], coords[i][1]),
                        (coords[i][0] + coords[i][2], coords[i][1] + coords[i][3]), (0, 0, 0), 2)
            cv.rectangle(resultImg, (coords[i][0], coords[i][1]),
                        (coords[i][0] + coords[i][2], coords[i][1] + coords[i][3]), textColor, 1)

            # put name
            cv.putText(img = resultImg, text = f"{predictionLabel[i]}", #{'%.3f'%(predictionPercent[i])}",
                                org = (coords[i][0], coords[i][1]), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.5, color = (0, 0, 0), thickness = 2)
            cv.putText(img = resultImg, text = f"{predictionLabel[i]}", #{'%.3f'%(predictionPercent[i])}",
                                org = (coords[i][0] + 1, coords[i][1] + 1), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.5, color = textColor, thickness = 1)
            # put coords
            cv.putText(img = resultImg, text = f"{(coords[i][0], coords[i][1])}",
                                org = (coords[i][0], coords[i][1] + 20), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.5, color = (0, 0, 0), thickness = 2)
            cv.putText(img = resultImg, text = f"{(coords[i][0], coords[i][1])}",
                                org = (coords[i][0] + 1, coords[i][1] + 21), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.5, color = textColor, thickness = 1)
            # put desired coords
            cv.putText(img = resultImg, text = f"{(desiredCoords[category][0], desiredCoords[category][1])}",
                                org = (coords[i][0], coords[i][1] + 40), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.5, color = (0, 0, 0), thickness = 2)
            cv.putText(img = resultImg, text = f"{(desiredCoords[category][0], desiredCoords[category][1])}",
                                org = (coords[i][0] + 1, coords[i][1] + 41), fontFace = cv.FONT_HERSHEY_TRIPLEX,
                                fontScale = 0.5, color = (0, 255, 255), thickness = 1)

        # Show in a window
        windowContest = np.concatenate((drawing, resultImg), axis=1)
        cv.imshow(windowName, windowContest)

        # set default coords
        #if (event == cv2.EVENT_LBUTTONDOWN):

    # edit desired item coords
    def on_mouse(self, a, x, y, press, empty):
        x = x - 640
        if (press == 1):
            print(self.itemCat)
            print(self.itemCoord)
            for i in range(len(self.itemCat)):
                if (x > self.itemCoord[i][0] and x < self.itemCoord[i][0] + self.itemCoord[i][2] and
                    y > self.itemCoord[i][1] and y < self.itemCoord[i][1] + self.itemCoord[i][3]):
                    desiredCoords[self.itemCat[i]] = [self.itemCoord[i][0], self.itemCoord[i][1]]
        print(x,y,press)

# contour deteciotn parameters
def empty(empty):
    pass

windowName = (f"VisionSystem    Loaded model: {model_path}    Categories: {items}")
cv.namedWindow(windowName)
cv.resizeWindow(windowName, 1200, 500)
cv.createTrackbar("Thresh_1", windowName, 80, 255, empty)
cv.createTrackbar("Thresh_2", windowName, 230, 500, empty)
cv.createTrackbar("AreaMin", windowName, 400, 30000, empty)
cv.createTrackbar("AreaMax", windowName, 12000, 30000, empty)


visionSystem = VisionSystem()

cv.setMouseCallback(windowName, visionSystem.on_mouse)

# start camera
webcam = cv.VideoCapture(0)

while True:
    button = False
    ok,frame = webcam.read()

    if ok == True:
        frameBlur = cv.GaussianBlur(frame, (7,7), 1)
        src_gray = cv.cvtColor(frameBlur, cv.COLOR_BGR2GRAY)

        tresh1 = cv.getTrackbarPos("Thresh_1", windowName)
        tresh2 = cv.getTrackbarPos("Thresh_2", windowName)
        
        visionSystem.detection(tresh1, tresh2)

        key = cv.waitKey(1)
        if(key == ord("q") or key == ord("z")):
            break
        
webcam.release()
cv.destroyAllWindows