import cv2

def createTracker(tracker_type):
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 4 and int(minor_ver) < 3:
        return cv2.cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            return cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            return cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            return cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            return cv2.TrackerMedianFlow_create()
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        if tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        else:
            raise ValueError('Invalid descriptor name')

def createDetector(detector_type):
    if detector_type == 'SIFT':
        return cv2.SIFT_create()
    elif detector_type == 'SURF':
        return cv2.xfeatures2d.SURF_create()
    elif detector_type == 'KAZE':
        return cv2.KAZE_create()
    elif detector_type == 'ORB':
        return cv2.ORB_create()
    elif detector_type == 'BRISK':
        return cv2.BRISK_create()
    elif detector_type == 'AKAZE':
        return cv2.AKAZE_create()
    else:
        raise ValueError('Invalid detector name')


def createDescriptor(descriptor_type):
    if descriptor_type == 'SIFT':
        return cv2.SIFT_create()       
    elif descriptor_type == 'SURF':
        return cv2.xfeatures2d.SURF_create()
    elif descriptor_type == 'KAZE':
        return cv2.KAZE_create()
    elif descriptor_type == 'BRIEF':
        return cv2.xfeatures2d.BriefDescriptorExtractor_create()
    elif descriptor_type == 'ORB':
        return cv2.ORB_create()
    elif descriptor_type == 'BRISK':
        return cv2.BRISK_create()
    elif descriptor_type == 'AKAZE':
        return cv2.AKAZE_create()        
    elif descriptor_type == 'FREAK':
        return cv2.xfeatures2d.FREAK_create()        
    else:
        raise ValueError('Invalid descriptor name')

def createMatcher(matcher_type, descriptor_type):
    if matcher_type == 'BF':
        if descriptor_type in ['SIFT', 'SURF', 'KAZE']:
            normType = cv2.NORM_L2
        else:
            normType = cv2.NORM_HAMMING

        return cv2.BFMatcher_create(normType=normType, crossCheck=True)
    elif matcher_type == 'FLANN':
        # Использование FLANN-based Matcher для сопоставления дескрипторов
        index_params = dict(algorithm=0, trees=5)  # Используем алгоритм KD-Tree
        search_params = dict(checks=50)  # Количество проверок для поиска
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError('Invalid matcher name')