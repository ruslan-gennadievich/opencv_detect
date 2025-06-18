import cv2
from feature_matcher import FeatureMatcher

FM = FeatureMatcher('SURF', 'SURF', 'FLANN')

img1 = cv2.imread('1_small.jpg')
img2 = cv2.imread('2_small.jpg')

FM.match(img1, img2)

coord = FM.get_target_coord()
if coord is not None:
    x, y = coord
    cv2.circle(img2, (x,y), 30, (0,255,0))
    cv2.imshow("Target", img2)

FM.plot_matches()

while cv2.waitKey(10) & 0xFF != ord('q'):
    pass
