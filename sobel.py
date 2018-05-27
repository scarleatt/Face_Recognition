import numpy as np
import cv2

image = cv2.imread("1.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite('sobel_1_source.jpg', image)

sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)
sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobelCombined = cv2.bitwise_or(sobelX, sobelY)
cv2.imwrite('sobel_1_sobelx.jpg', sobelX)
cv2.imwrite('sobel_1_sobely.jpg', sobelY)
cv2.imwrite('sobel_1_sobel_combined.jpg', sobelCombined)

# cv2.imshow("Sobel X", sobelX)
# cv2.waitKey()
# cv2.imshow("Sobel Y", sobelY)
# cv2.waitKey()
# cv2.imshow("Sobel Combined", sobelCombined)

# cv2.waitKey()