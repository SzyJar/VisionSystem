import cv2
import numpy as np
import matplotlib.pyplot as plt

template = cv2.imread('item.jpg')
template_img = template
plt.axis('off')
plt.imshow(template)
plt.title('TEMPLATE')
plt.show()

template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

target = cv2.imread('scene.jpg')
target_gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
plt.axis('off')
plt.imshow(target)
plt.title('TARGET')
plt.show()

ret,thresh1 = cv2.threshold(template,127,255,0)
ret,thresh2 = cv2.threshold(target_gray,127,255,0)

contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(template_img,contours,-1,(0,255,0),3)
plt.axis('off')
plt.imshow(template_img)
plt.title('Identifying the Contours')
plt.show()

#SORT THE CONTOURS BY AREA
# doing this so we can remove the largest  contour which is the image outline

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

#extract the second largest contour which is the shape only
template_contour = contours[1]

contours,hierarchy = cv2.findContours(thresh2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    match = cv2.matchShapes(template_contour,c,3,0.0)
    print(match)
    #valid matches would be less than 0.15
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []

cv2.drawContours(target,[closest_contour],-1,(0,255,0),3)
plt.axis('off')
plt.imshow(target)
plt.title('SUCCESSFULLY MATCHED SHAPE')
plt.show()