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

# #순서대로 아래, 위(아래가 양수(100만큼)), 오른쪽 왼쪽
# move = np.float32([[1, 0, 100], [0, 1, 100]])
# moved = cv2.warpAffine(img, move, (width, height))
# cv2.imshow("Moved down: +, up: - and right: +, left -", moved)

# #rotate함수
# #중심값을 기준으로 각도만큼(90도clokwise), scale
# move = cv2.getRotationMatrix2D(center, -90, 1)
# rotated = cv2.warpAffine(img, move, (width, height))
# cv2.imshow("Rotated clokwise degrees", rotated)

# #사이즈를 변형
# ratio = 200.0 / width
# dimension = (200, int(height * ratio)) #X, Y값
# resized = cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)
# cv2. imshow("Resized", resized)

# #flip
# flipped = cv2.flip(img, 1)
# cv2.imshow("Flipped Horizontal 1, Vertical 0, both -1", flipped)


cv2.waitKey(0)
cv2.destroyAllWindows()