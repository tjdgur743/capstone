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

#RGB 채널 분리
(Blue, Green, Red) = cv2.split(img)

cv2.imshow("Red Channel", Red)
cv2.imshow("Green Channel", Green)
cv2.imshow("Blue Channel", Blue)
cv2.waitKey(0)

#직관적으로 보기 위해 RGB에 해당하는 색 외에는 검정색으로
zeros = np.zeros(img.shape[:2], dtype="uint8")
cv2.imshow("Red Channel", cv2.merge([zeros, zeros, Red]))
cv2.imshow("Green Channel", cv2.merge([zeros, Green, zeros]))
cv2.imshow("Blue Channel", cv2.merge([Blue, zeros, zeros]))
cv2.waitKey(0)

#Filter
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", gray)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB", lab)
cv2.waitKey(0)

#채널 분리한 것들을 합치면 원래의 이미지와 같다.
BGR = cv2.merge([Blue, Green, Red])
cv2.imshow("Merge", BGR)

cv2.waitKey(0)
cv2.destroyAllWindows()