# Sobel算子
import numpy as np
from skimage import data
import matplotlib.pyplot as plt
import cv2
#image = data.camera()
image = cv2.imread("../testImage/img3.jpg", 0)


def SobelOperator(roi, operator_type):
    if operator_type == "horizontal":
        sobel_operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator_type == "vertical":
        sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        raise ("type Error")
    result = np.abs(np.sum(roi * sobel_operator))
    return result


def SobelAlogrithm(image, operator_type):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = SobelOperator(image[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(image))
    return new_image.astype(np.uint8)


plt.figure('thresh', figsize=(8, 8))
plt.subplot(121)
plt.title("Original")
plt.imshow(image, plt.cm.gray)
plt.axis("off")
plt.subplot(122)
plt.title("vertical")
plt.imshow(SobelAlogrithm(image, "vertical"), cmap="binary")
plt.axis("off")
plt.show()
