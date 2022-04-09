import cv2

img = cv2.imread("aaa.png")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))

cv2.imshow("aaa", img)

(b, g, r) = img[0, 0]
print("Pixel at (0,0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

#img의 50:100, 50:100범위 저장
#dot = img[50:100, 50:100]
#cv2.imshow("Dot", dot)

#해당 범위의 b,g,r 의 값을 준다.
img[50:100, 50:100] = (0, 0, 255)

#사진, 시작점, 끝점, b,g,r, 선의 굵기
cv2.rectangle(img, (150, 50), (200, 100), (0, 255, 0), 5)
#중심의 위치, 반지름 크기, b,gㅂ,r, 공간을 체운다
cv2.circle(img, (275, 75), 25, (0, 255, 255), -1)
#시작점, 끝점, b,g,r, 선의 굵기
cv2.line(img, (350, 100), (400, 100), (255, 0, 0), 5)
#글씨를 그린다.
#쓸 글씨, 시작점, 폰트지정, 폰트의 크기, b,g,r, 폰트의 굵기
cv2.putText(img, "creApple", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 4)
cv2.imshow("aaa-dotted", img)

cv2.waitKey(0)
cv2.destroyAllWindows()