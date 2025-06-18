"""Import opencv"""
import cv2
from color_processing import LAB_Color_Space
from feature_matcher import FeatureMatcher

class TargetTracker:
    __TargetImage_Original = None
    __TargetImageColorCorrected = None
    __TargetKeypoints = None
    __TargetDescriptors = None
    __TargetWidth = 0
    __TargetHeight = 0
    __TargetCoordX = 0
    __TargetCoordY = 0
    __Mode = None #tracking/detecting
    __FeatureMatcher = None
    LastInputFrameImage = None
    LastOutputFrameImage = None
    
    DefaultDetector = 'SIFT'    #See FeatureMatcher
    DefaultDescriptor = 'SIFT'  #See FeatureMatcher
    DefaultMatcher = 'FLANN'    #See FeatureMatcher

    def __init__(self):
        self.__FeatureMatcher = FeatureMatcher(self.DefaultDetector, self.DefaultDescriptor, self.DefaultMatcher)

    def set_target(self, image, process_color=True):
        if process_color:
            image = self.process_color(image)


    def new_frame(self, new_image_frame):
        """Processed new frame"""
        self.LastInputFrameImage = new_image_frame
        self.__TargetImageColorCorrected = self.process_color(new_image_frame)

    def get_rectangle_coord(self):
        """Get coords for rectangle of target"""
        return (self.__TargetCoordX - int(self.__TargetWidth / 2),
                self.__TargetCoordY - int(self.__TargetHeight / 2),
                self.__TargetWidth,
                self.__TargetHeight)

    def get_target_coord(self):
        """Get coords (X, Y) center of target"""
        return self.__TargetCoordX, self.__TargetCoordY

    def process_color(self, img):
        """Color corection"""
        #return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)
        #return LAB_Color_Space(img)
