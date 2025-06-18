import cv2
import numpy as np

# Функция для предварительной обработки изображений
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Применение гауссового размытия для снижения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# Загрузка изображений
image1 = cv2.imread("image1.jpg")
image2 = cv2.imread("image2.jpg")

# Предварительная обработка изображений
image1_gray = preprocess_image(image1)
image2_gray = preprocess_image(image2)

# Выбор региона интереса (ROI) на первом изображении
roi = cv2.selectROI("Select ROI", image1, fromCenter=False, showCrosshair=True)

# Обрезка выделенной области на первом изображении
roi_cropped = image1[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
roi_cropped_gray = preprocess_image(roi_cropped)

# Использование SIFT детектора для поиска ключевых точек и дескрипторов
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(roi_cropped_gray, None)
kp2, des2 = sift.detectAndCompute(image2_gray, None)

# Использование FLANN-based Matcher для сопоставления дескрипторов
index_params = dict(algorithm=0, trees=5)  # Используем алгоритм KD-Tree
search_params = dict(checks=50)  # Количество проверок для поиска
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Применение Ratio Test для фильтрации совпадений
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


# Если достаточно хороших совпадений
if len(good_matches) > 3:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Используем гомографию для поиска области на втором изображении
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is not None:
        h, w = roi_cropped_gray.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Рисуем рамку на найденном объекте на втором изображении
        image2 = cv2.polylines(image2, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        
        #image2_h, image2_w = image2_gray.shape
        #image2 = cv2.warpPerspective(image2, dst, (image2_w,image2_h))
    else:
        print("Не удалось найти гомографию")

# Рисуем хорошие совпадения на изображениях
img_matches = cv2.drawMatches(roi_cropped, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Отображаем результат
cv2.imshow("Matches", img_matches)
cv2.imshow("Detected", image2)

cv2.waitKey(0)

cv2.destroyAllWindows()
