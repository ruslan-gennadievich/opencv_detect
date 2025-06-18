import cv2
from edges_matcher import EdgesMatcher

template = cv2.imread('image1.jpg')
img1 = cv2.imread('image2.jpg')
em = EdgesMatcher()

roi = cv2.selectROI("Edgte template", template, fromCenter=False, showCrosshair=True)
if roi[2] > 0 and roi[3] > 0:
    template = template[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    
em.load_template_image(template)

top_left, bottom_right, best_scale = em.match_edges(img1)

cv2.rectangle(img1, top_left, bottom_right, 255, 2)
cv2.putText(img1, 'best_scale:' + str(best_scale), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, lineType=cv2.LINE_AA)

cv2.imshow("Edgte template", img1)
cv2.waitKey()
cv2.destroyWindow("Edgte template")