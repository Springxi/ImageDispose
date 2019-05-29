import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../testImage/img1.jpg", 0)
#归一化
fi = img/255.0
gamma = 0.4
out = np.power(fi, gamma)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()