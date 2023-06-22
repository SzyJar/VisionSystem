import cv2 as cv
import matplotlib.pyplot as plt


def brute():
    img1 = cv.imread('./Images/pattern/strip.jpg',cv.IMREAD_GRAYSCALE) # QueryImage
    img2 = cv.imread('./Images/pattern/strip_rot.jpg',cv.IMREAD_GRAYSCALE) # TrainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    print(len(good))
    plt.imshow(img3),plt.show()


webcam = cv.VideoCapture(0)

ok,frame = webcam.read()

if ok == True:
    brute()

webcam.release()
cv.destroyAllWindows