import cv2
import numpy as np
from matplotlib import pyplot as plt

# Загружаем два изображения
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)  # Image 1
img2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)  # Image 2

# Используем ORB для поиска ключевых точек и вычисления дескрипторов
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Сопоставление дескрипторов с использованием BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Выбираем только лучшие 10 совпадений
good_matches = matches[:10]

# Вычисляем гомографию
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Трансформируем первое изображение
h, w = img2.shape
img1_aligned = cv2.warpPerspective(img1, H, (w, h))

# Определение рамки (прямоугольника) вокруг объекта на первом изображении
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, H)

# Преобразование img1 в цветное изображение для отображения рамки
img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img1_aligned_color = cv2.cvtColor(img1_aligned, cv2.COLOR_GRAY2BGR)

# Рисуем рамку на обоих изображениях
img1_with_box = cv2.polylines(img1_color, [np.int32(pts)], True, (0, 255, 0), 3, cv2.LINE_AA)
img2_with_box = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
img1_aligned_with_box = cv2.polylines(img1_aligned_color, [np.int32(pts)], True, (0, 255, 0), 3, cv2.LINE_AA)

# Визуализация результатов
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Image 1 with Box')
plt.imshow(cv2.cvtColor(img1_with_box, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Aligned Image 1 with Box')
plt.imshow(cv2.cvtColor(img1_aligned_with_box, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title('Image 2 with Box')
plt.imshow(cv2.cvtColor(img2_with_box, cv2.COLOR_BGR2RGB))

plt.show()
