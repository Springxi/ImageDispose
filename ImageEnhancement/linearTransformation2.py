#分段线性变换
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 绘制直方图函数
def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBin = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBin,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()
    cv.imshow("img", img)


img = cv.imread("../testImage/img1.jpg", 0)
img = cv.resize(img, None, fx=0.95, fy=0.95)   #沿x，y轴缩放倍数
h, w = img.shape[:2]
out = np.zeros(img.shape, np.uint8)
###################################################这里的参数需要调！！！！！！！！！！！！！！
for i in range(h):
    for j in range(w):
        pix = img[i][j]
        if pix < 50:
            out[i][j] = 0.5 * pix
        elif pix < 150:
            out[i][j] = 3.6 * pix - 310
        else:
            out[i][j] = 0.238 * pix + 104
out[out > 255] = 255    #进行数据截断，大于255的值截断为255
out = np.around(out)    #强制类型转换，numpy.around(arr, decimals=0, out=None)
out = out.astype(np.uint8) #编码格式,无符号8位int

#绘制直方图
grayHist(img)
grayHist(out)
cv.imshow("img", img)
cv.imshow("out", out)

cv.waitKey()