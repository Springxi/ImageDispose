import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#计算灰度直方图
def calcGraHist(I):
    h, w = I.shape[:2]   #取彩色图片的长、宽,I.shape[:3] 取取彩色图片的长、宽、通道
    grayHist = np.zeros([256], dtype=np.uint64)   #创建一个一维数组grayHist，长度为255，用其序列表示灰度值
    for i in range(h):
        for j in range(w):
            # 遍历所有元素，把其灰度值所代表的序列指向的数组grayHist累加
            grayHist[I[i][j]] += 1
    return grayHist

img = cv.imread("../testImage/img2.jpg", 0)    #图像的读取
grayHist = calcGraHist(img)
x = np.arange(256)   #np.arange()函数返回一个有终点和起点的固定步长的排列

#绘制灰度直方图
plt.plot(x, grayHist, 'r', linewidth=2, c='black')
plt.xlabel("gray Label")
plt.ylabel("number of pixels")
plt.show()
cv.imshow("img", img)    #图像的载入
cv.waitKey()    #保持，没有的话在IDLE中执行窗口直接无响应。在命令行中执行的话，则是一闪而过

