import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#计算灰度直方图
def calcGraHist(I):
    h, w = I.shape[:2]   #取彩色图片的长、宽,I.shape[:3] 取取彩色图片的长、宽、通道
    grayHist = np.zeros([256], dtype=np.uint32)   #创建一个一维数组grayHist，长度为255，用其序列表示灰度值
    for i in range(h):
        for j in range(w):
            # 遍历所有元素，把其灰度值所代表的序列指向的数组grayHist累加
            grayHist[I[i][j]] += 1
    return grayHist


#画灰度直方图
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


#全局直方图均衡化
def euqalHist(img):

    h, w = img.shape[:2]

    # 1、计算灰度直方图
    grayHist = calcGraHist(img)
    print(grayHist)
    # 2、计算累加灰度直方图（统计的总的个数）
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    print(zeroCumuMoment)
    #3、根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0/(h * w)
    #sss = 'h:'+ str(h) +'  w:'+ str(w) +'coff:'+ str(cofficient)

    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if p >= 0:
            outPut_q[p] = np.math.floor(q)   #返回数字的下舍整数
        else:
            outPut_q[p] = 0
    #4、得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage




img = cv.imread("../testImage/img2.jpg", 0)
#使用自己写的函数实现
out = euqalHist(img)

#使用opencv的函数实现
#out = cv.equalizeHist(img)

grayHist(img)
grayHist(out)

cv.imshow("img", img)
cv.imshow("out", out)
cv.waitKey()