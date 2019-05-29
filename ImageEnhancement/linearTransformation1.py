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
out = 1.3 * img
out[out > 255] = 255    #进行数据截断，大于255的值截断为255
out = np.around(out)
#强制类型转换，numpy.around(arr, decimals=0, out=None)
#decimals为n对输入近似后保留小数点后n位，默认为0，若值为-n，则对小数点左边第n位近似；
out = out.astype(np.uint8) #编码格式,无符号8位int

#绘制直方图
grayHist(img)
grayHist(out)
cv.imshow("img", img)
cv.imshow("out", out)

cv.waitKey()