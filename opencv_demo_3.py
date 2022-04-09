from audioop import ratecv
import cv2
import numpy as np

img = cv2.imread("aaa.png")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

(height, width) = img.shape[:2]
center = (width//2, height//2)

cv2.imshow("aaa", img)

#높이 = 0, 넓이 = 1 -> 0으로
mask = np.zeros(img.shape[:2], dtype = "uint8")

cv2.circle(mask, center, 200, (255, 255, 255), -1)

cv2.imshow("mask", mask)

masked = cv2.bitwise_and(img, img, mask = mask)
cv2.imshow("masked aaa", masked)

cv2.waitKey(0)
cv2.destroyAllWindows()