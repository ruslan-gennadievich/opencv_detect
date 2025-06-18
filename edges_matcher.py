import cv2
from color_processing import LAB_Color_Space

class EdgesMatcher:
    _scales = [1.3, 1.2, 1.1, 0.9, 0.8, 0.7]
    _templatesScaledEdges = []
    
    def load_template_image(self, image, scale=True):
        image = self._processColor(image)
        if scale is True:
            for scale_coff in self._scales:
                scaled_image = cv2.resize(image, (0, 0), fx=scale_coff, fy=scale_coff)
                scaled_edges = cv2.Canny(scaled_image, 50, 150)
                self._templatesScaledEdges.append((scaled_image, scaled_edges, scale_coff))
        else:
            image_edges = cv2.Canny(image, 50, 150)
            self._templatesScaledEdges.append((image_edges, 1.0))


    def match_edges(self, frame):
        frame = self._processColor(frame)
        img2 = None
        frame_edges = cv2.Canny(frame, 50, 150)
        
        best_max_val = -1
        best_loc = None
        best_scale = None
        for templateScaledEdges in self._templatesScaledEdges:
            scaled_image, scaled_edges, scale_coff = (templateScaledEdges)
            result = cv2.matchTemplate(frame_edges, scaled_edges, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val > best_max_val:
                best_max_val = max_val
                best_loc = max_loc
                best_scale = scale_coff

        top_left = best_loc
        #top_left = (int(top_left[0] + scaled_image.shape[1] / 4), int(top_left[1] + scaled_image.shape[0] / 4))

        bottom_right = (int(top_left[0] + scaled_image.shape[1] * best_scale), int(top_left[1] + scaled_image.shape[0] * best_scale))
        return top_left, bottom_right, best_scale
        
    def _processColor(self, img):        
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.equalizeHist(img)
        #clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        #img = clahe.apply(img)
        #return cv2.GaussianBlur(img, (7, 7), 0)
        #return LAB_Color_Space(img)
        return img