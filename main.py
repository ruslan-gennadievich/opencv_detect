import cv2
import numpy as np
import time
import os
from copy import copy
from mat_helpers import cluster_points, compute_centroids
from color_processing import LAB_Color_Space
from factory import createDetector, createDescriptor, createMatcher, createTracker
from web_server import ImageUploadServer

## === Вспомогательные функции === ###
def processColor(img):
    #return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)
    #return LAB_Color_Space(img)

def get_features(image, detectorType, descriptorType): # detect keypoints and compute descriptors
    if detectorType == descriptorType or (detectorType == 'SIFT' and descriptorType == 'SIFT'):        
        keypoints, descriptors = DETECTOR.detectAndCompute(image, None)
    else:
        keypoints = DETECTOR.detect(image, None)
        keypoints, descriptors = DESCRIPTOR.compute(image, keypoints)
    return keypoints, descriptors
## ===
     
#cap = cv2.VideoCapture('libcamerasrc ! video/x-raw, width=640, height=480, framerate=25/1 ! videoconvert ! appsink drop=true', cv2.CAP_GSTREAMER) # Raspberry
cap = cv2.VideoCapture(2) # Notebook
#cap = cv2.VideoCapture('IMG_8491.MOV') # Video

# Проверка, что OpenCV скомпилирован с поддержкой GStreamer
if cv2.getBuildInformation().find('GStreamer:                   NO') > -1:
    print('ERROR: Your OpenCV not support GStreamer')
    exit(1)

# Проверка, что камера успешно открылась
if not cap.isOpened():
    print("Ошибка открытия камеры!")
    exit(2)

webserver = ImageUploadServer()
webserver.start()
detectorType = 'SIFT' # See factory.py to know avaible
descriptorType = 'SIFT'
matcher = 'FLANN' # BF / FLANN

DETECTOR = createDetector(detectorType)
DESCRIPTOR = createDetector(descriptorType)
MATCHER = createMatcher(matcher, descriptorType) 
TRACKER = createTracker('KCF')

showCenterPoint = True # Определять центральную точку по найденным ключевым точкам (без этого не будет работать трекер!)
showHomography = False # Соединять похожие точки на образце и кадре линиями (гомография, без этого трекер работает)

def referenceLoad(image):
    global Mode, reference_colorProcessed, reference_keypoints, reference_descriptors, reference_height, reference_width, detectorType, descriptorType

    Mode = 'Detect' # Detect \ Track \ Wait to select
    
    # Корекция цвета эталонного изображения
    reference_colorProcessed = processColor(image)
    
    reference_image = image
    #reference_image = reference_colorProcessed # Чтобы отображалось то изображение, которое обработали (так как точки находим на reference_colorProcessed)

    reference_keypoints, reference_descriptors = get_features(reference_colorProcessed, detectorType, descriptorType)
    reference_height, reference_width,c  = reference_image.shape
    print('reference loaded')


# Загрузка эталонного изображения
if os.path.isfile('cropped_image.jpg'):
    reference_image = cv2.imread('cropped_image.jpg')
    if reference_image is None:
        print("Ошибка загрузки эталонного изображения!")
        exit(3)
    referenceLoad(reference_image)
else:
    Mode = 'Wait to select'

prev_frame_time = 0

print ("Mode: " + Mode)
RatioTestMatchesCoff = 0.8
while True: # Захват кадра из веб-камеры
    ret, frame = cap.read() #frame - чистый фрейм который передается в обработку
    if ret == False:
        break

    finishFrame = copy(frame) # фрейм поверх которого отображается информация (буквы, линии) и этот фрейм выводится пользователю
    new_frame_time = time.time()
    key = cv2.waitKey(1) & 0xFF

    if webserver.uploaded_CV_Image is not None:
        referenceLoad(webserver.uploaded_CV_Image)
        cv2.imwrite("cropped_image.jpg", webserver.uploaded_CV_Image)
        webserver.uploaded_CV_Image = None
    
    if Mode == 'Detect':        
        if not ret:
            print("Не удалось захватить кадр!")
            break
        
        # Корекция цвета текущего кадра
        frameColorProcessed = processColor(frame)
        #frame = frameColorProcessed # Чтобы отображалось то изображение, которое обработали (так как точки находим на frameColorProcessed)
                
        # Детектирование ключевых точек и вычисление дескрипторов
        keypoints, descriptors = get_features(frameColorProcessed, detectorType, descriptorType)
        
        if descriptors is not None and reference_descriptors is not None:  # Если дескрипторы были найдены
            # Сопоставление дескрипторов
            if detectorType == 'SIFT' and descriptorType == 'SIFT' and matcher != 'BF':
                matches = MATCHER.knnMatch(descriptors, reference_descriptors, k=2)
                # Применение Ratio Test для фильтрации совпадений
                good_matches = []
                for m, n in matches:
                    if m.distance < RatioTestMatchesCoff * n.distance:
                        good_matches.append(m)
                matches = good_matches
                if len(matches) > 6 and RatioTestMatchesCoff > 0.3:
                    RatioTestMatchesCoff = RatioTestMatchesCoff - 0.02                    
                elif len(matches) < 6 and RatioTestMatchesCoff < 0.8:
                    RatioTestMatchesCoff = RatioTestMatchesCoff + 0.02                    

                print ('RatioTestMatchesCoff:' + str(RatioTestMatchesCoff))
            else:
                matches = MATCHER.match(descriptors, reference_descriptors)            
                # Сортировка по качеству совпадений
                matches = sorted(matches, key=lambda x: x.distance)
                matches = matches[:10] # Limits the number of matches to be displayed
            
            # Извлечение точек совпадений
            src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([reference_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            # Нахождение гомографии
            matches_mask = None
            if showHomography == True:                
                if len(src_pts) >= 4 and len(dst_pts) >= 4:  # Нужно хотя бы 4 точки для нахождения гомографии
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    matches_mask = mask.ravel().tolist()

            if showCenterPoint is True and len(matches) > 1:
                # Кластеризация точек (нахождение кластеров точек находящихся рядом)
                radius = int(min(reference_width, reference_height) * 1.3)
                
                matchesInCluster = int(len(matches) / 4)
                if matchesInCluster < 3:
                    matchesInCluster = 3
                print(matchesInCluster)
                clusters = cluster_points(src_pts, radius, matchesInCluster)
                

                #Если есть хотябы один кластер и внутри него есть больше двух точек рядом
                if len(clusters) > 0 and len(clusters[0]) > 0:
                    
                    # Нахождение центроидов кластеров (нахождение средней точки внутри кластера)
                    centroids = compute_centroids(clusters, src_pts)            
                    for centroid in centroids:                
                        x, y = map(int, centroid.astype(int)[0]) # КООРДИНАТЫ ЦЕЛИ !                    
                        cv2.line(frame, (x, y-30), (x, y+30), (255, 255, 255), 2) # Мишень (крестик)
                        cv2.line(frame, (x-30, y), (x+30, y), (255, 255, 255), 2) # Мишень (крестик)                        
                        finishFrame = cv2.rectangle(finishFrame, (x-30, y-30, reference_width, reference_height), (255, 0, 0), 2) # Рамка
                        
                        TRACKER = createTracker('KCF')
                        TRACKER.init(frame, (x-30, y-30, reference_width, reference_height)) # Эту же "рамку" передаем в трекер
                        Mode = 'Track' # Переключаем режим
                        print ("Mode: " + Mode)

            # Отрисовка совпадений            
            finishFrame = cv2.drawMatches(frame, keypoints, reference_image, reference_keypoints, matches, None, matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        else:
            print ("Descriptors or reference_descriptors not found!")
            #key = ord("c")
    elif Mode == 'Track':        
        (success, box) = TRACKER.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            finishFrame = copy(frame)
            cv2.rectangle(finishFrame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        else:            
            Mode = 'Detect'
            print ("Mode: " + Mode)

    if key == ord("c"): # Выбор нового изображения в качестве цели при нажатии клавиши 'c'
        r = cv2.selectROI("Keypoints and Matches", frame, fromCenter=False, showCrosshair=False)        
        if r[2] == 0 or r[3] == 0:
            print("Выделенная область пуста!")
            continue
        reference_width, reference_height = r[2], r[3]

        # Извлечение выбранной области
        x, y, w, h = map(int, r)
        reference_image = frame[y:y+h, x:x+w]
        cv2.imwrite("cropped_image.jpg", reference_image)
        Mode = 'Detect'
        try:            
            reference_colorProcessed = processColor(reference_image)
        except:
            reference_colorProcessed = reference_image
        
        reference_keypoints, reference_descriptors = get_features(reference_colorProcessed, detectorType, descriptorType)
    
    if key == ord("q"): # Выход из цикла при нажатии клавиши 'q'
        break

    
    fps = str(int(1/(new_frame_time-prev_frame_time)))
    prev_frame_time = new_frame_time

    finishFrame = cv2.putText(finishFrame, ' MODE: ' + Mode + ' FPS: ' + fps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('Keypoints and Matches', finishFrame)
    

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
