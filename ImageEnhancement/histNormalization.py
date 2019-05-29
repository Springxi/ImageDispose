import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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
Imin, Imax = cv.minMaxLoc(img)[:2]
#minVal，maxVal，minLoc，maxLoc = cv.minMaxLoc（src [，mask]）
#返回值分别为：最小值，最大值，最小值的位置索引，最大值的位置索引。
Omin, Omax = 0, 255

a = float(Omax - Omin) / (Imax - Imin)
b = Omin - a * Imin
out = a * img + b
out = out.astype(np.uint8)

#等价于————
#cv.normalize(img, out, 255, 0, cv.NORM_MINMAX, cv.CV_8U)


#正规化函数normalize:
# dst=cv.normalize(src, dst, alpha, beta, norm_type[], dtype, mask[])
#src  输入数组；
#dst  输出数组，数组的大小和原数组一致；
#alpha Omix；
#beta  Omin;
#norm_type   归一化选择的数学公式类型；
#               {NORM_L1(Ai/Ai的和), NORM_INF（Ai/max(Ai)）, NORM_L2（Ai/aqrt(Ai的和)）, NORM_MINMAX  (Ai/（max(Ai)-min(Ai)）)}；
#dtype     当为负，输出在大小深度通道数都等于输入，当为正，输出只在深度与输如不同，不同的地方游dtype决定；
#               一般的图像文件格式使用的是 Unsigned 8bits吧，CvMat矩阵对应的参数类型就是CV_8UC1，CV_8UC2，CV_8UC3,123代表通道数
#mark      掩码。选择感兴趣区域，选定后只能对该区域进行操作。


grayHist(img)
grayHist(out)
cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()