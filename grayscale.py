import cv2
import matplotlib.pyplot as plt

image = cv2.imread('./milkyway.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imwrite('./milkyway_gray.jpg', image)

plt.figure('milkyway')
plt.imshow(image, cmap='gray')  # cmap即colormap，颜色映射
plt.axis('off')
plt.show()