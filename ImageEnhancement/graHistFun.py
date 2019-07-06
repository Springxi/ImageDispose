import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("../testImage/img3.jpg", 0)
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
cv.waitKey()

#matplotlib.pyplot.hist()函数_
#返回值    histogram（直方图向量，是否归一化由参数normed设定）,
#          bins,（各个bin的区间范围）
#          patch（每个bin里面包含的数据，是一个list）
#参数   pixelSequence: 需要计算直方图的一维数组
#       bins: 直方图的柱数，可选项，默认为10
#       normed: 是否将得到的直方图向量归一化。默认为0
#       facecolor: 直方图颜色
#       edgecolor: 直方图边框颜色
#       alpha: 透明度
#       histtype : {‘bar’, ‘barstacked’, ‘step’, ‘stepfilled’},
