
import numpy as np 
import cv2
import imutils 
from skimage.filters import threshold_local
from imutils.perspective import four_point_transform

image= cv2.imread("/content/drive/MyDrive/img.jpg")
ratio= image.shape[0]/500.0
orig= image.copy()
#resizing 
image= imutils.resize(image, height=500)

gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
smooth= cv2.GaussianBlur(gray, (5,5), 0)
edged= cv2.Canny(gray, 75, 200)

cnts= cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts= imutils.grab_contours(cnts)
cnts= sorted(cnts, key= cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri= cv2.arcLength(c, True)
    approx= cv2.approxPolyDP(c, 0.02*peri, True)
    if(len(approx)== 4):
        screenCnt= approx
        break

warped= four_point_transform(orig, screenCnt.reshape(4,2)*ratio)

from google.colab.patches import cv2_imshow
#cv2_imshow(image)
#cv2_imshow(edged)
cv2_imshow(warped)
cv2.waitKey(0)