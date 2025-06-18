import cv2
import numpy as np

def preprocess_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def match_edges(template, frame):
    template_edges = cv2.Canny(template, 50, 150)
    frame_edges = cv2.Canny(frame, 50, 150)

    result = cv2.matchTemplate(frame_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    return max_loc, max_val

def match_edges_multi_scale(template, frame, scales=[1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]):
    max_val = -np.inf
    best_loc = None
    best_scale = None

    for scale in scales:
        scaled_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        max_loc, value = match_edges(scaled_template, frame)

        if value > max_val:
            max_val = value
            best_loc = max_loc
            best_scale = scale

    return best_loc, max_val, best_scale

#cap = cv2.VideoCapture(2)
cap = cv2.VideoCapture('IMG_8491.MOV') 
template = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = preprocess_image(frame)

    # Если шаблон уже задан, выполняем сопоставление
    if template is not None:
        max_loc, max_val, best_scale = match_edges_multi_scale(template, frame)
        top_left = max_loc
        bottom_right = (int(top_left[0] + template.shape[1] * best_scale), int(top_left[1] + template.shape[0] * best_scale))
        cv2.rectangle(frame, top_left, bottom_right, 255, 2)

    cv2.imshow("Edgte template", frame)

    # Обработка нажатия клавиш
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Захват текущего кадра для выбора ROI
        roi = cv2.selectROI("Edgte template", frame, fromCenter=False, showCrosshair=True)
        if roi[2] > 0 and roi[3] > 0:
            template = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            template = preprocess_image(template)
        cv2.destroyWindow("Edgte template")

cap.release()
cv2.destroyAllWindows()
