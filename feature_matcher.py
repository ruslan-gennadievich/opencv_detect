"""Feature Description and Matching"""
from copy import copy
import numpy as np
import cv2
from matplotlib import pyplot as plt

class FeatureMatcher:
    BF_LIMIT = 30           # Limits the number of matches to be displayed
    FLANN_INDEX_KDTREE = 1  # FLANN INDEX KDTREE parameter
    FLANN_RATIO_TH = 0.7    # Limits the number of matches to be displayed
    FLANN_CHECKS = 50       # higher: more accurate, but slower
    HOMOGRAPHY_MATCH_TH = 10  # number of matches that are necessary for homography
    HOMOGRAPHY_RANSAC_TH = 5  # Maximum reprojection error in the RANSAC algorithm to consider a point as an inlier.

    def __init__(self, detector_name, descriptor_name, matcher_name):
        """
        :param detector_name: (SIFT, SURF, KAZE, ORB, BRISK, AKAZE)
        :param descriptor_name: (SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK)
        :param matcher_name: (BF, FLANN)

        Attributes:
            self.detector: detector object (SIFT, SURF, KAZE, ORB, BRISK, AKAZE)
            self.descriptor: descriptor object (SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK)
            self.matcher: matcher (Brute Force Matcher or FlannBasedMatcher)
        """
        # names
        self.detector_name = detector_name
        self.descriptor_name = descriptor_name
        self.matcher_name = matcher_name
        # objects
        self.detector = self._get_detector()
        self.descriptor = self._get_descriptor()
        # images to match
        self.img1 = None
        self.img2 = None
        # keypoints and descriptors
        self.kpt1, self.des1 = None, None
        self.kpt2, self.des2 = None, None
        # resulting matches
        self.matches = []
        self.matches_img = None
        self.time = -1.0
        self.kpt1_src_pts = None
        self.kpt2_dst_pts = None

    def _get_detector(self):
        """
        Uses the detector name to return the detector object
        :return: detector object (SIFT, SURF, KAZE, ORB, BRISK, AKAZE)
        """
        if self.detector_name == 'SIFT':
            return cv2.xfeatures2d.SIFT_create()
        elif self.detector_name == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        elif self.detector_name == 'KAZE':
            return cv2.KAZE_create()
        elif self.detector_name == 'ORB':
            return cv2.ORB_create()
        elif self.detector_name == 'BRISK':
            return cv2.BRISK_create()
        elif self.detector_name == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError('Invalid detector name')

    def _get_descriptor(self):
        """
        Uses the descriptor name to return the descriptor object
        :return: descriptor object (SIFT, SURF, KAZE, BRIEF, ORB, BRISK, AKAZE, FREAK)
        """
        if self.descriptor_name == 'SIFT':
            return cv2.xfeatures2d.SIFT_create()
        elif self.descriptor_name == 'SURF':
            return cv2.xfeatures2d.SURF_create()
        elif self.descriptor_name == 'KAZE':
            return cv2.KAZE_create()
        elif self.descriptor_name == 'BRIEF':
            return cv2.xfeatures2d.BriefDescriptorExtractor_create()
        elif self.descriptor_name == 'ORB':
            return cv2.ORB_create()
        elif self.descriptor_name == 'BRISK':
            return cv2.BRISK_create()
        elif self.descriptor_name == 'AKAZE':
            return cv2.AKAZE_create()
        elif self.descriptor_name == 'FREAK':
            return cv2.xfeatures2d.FREAK_create()
        else:
            raise ValueError('Invalid descriptor name')

    def _get_features(self, image):
        """
        :param image: input image
        :return: keypoints, descriptors
        """
        # detect keypoints and compute descriptors
        if self.detector_name == self.descriptor_name:
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
        else:
            keypoints = self.detector.detect(image, None)
            keypoints, descriptors = self.descriptor.compute(image, keypoints)
        return keypoints, descriptors

    def match(self, image1, image2, DYNAMIC_FLANN_RATIO_TH = True, equalize=False):
        """
        :param image1: input image 1
        :param image2: input image 2
        :param equalize: if True, equalizes the images before matching
        :return: matches
        """
        # measure time
        start = cv2.getTickCount()

        # histogram equalization
        if equalize:
            if len(image1.shape) != 2:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            if len(image2.shape) != 2:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            
            image1 = cv2.equalizeHist(image1)            
            image2 = cv2.equalizeHist(image2)

        self.img1 = copy(image1)
        self.img2 = copy(image2)

        # get features
        self.kpt1, self.des1 = self._get_features(self.img1)
        self.kpt2, self.des2 = self._get_features(self.img2)

        # match features
        if self.matcher_name == 'BF':
            # if descriptor_name is SIFT, SURF or KAZE, set normType=cv2.NORM_L2
            if self.descriptor_name in ['SIFT', 'SURF', 'KAZE']:
                normType = cv2.NORM_L2
            else:
                normType = cv2.NORM_HAMMING

            # brute force matching
            bf = cv2.BFMatcher_create(normType=normType, crossCheck=True)  #TODO: why crossCheck=True?
            matches_all = bf.match(self.des1, self.des2)
            # sort matches_img in the order of their distance
            matches_all = sorted(matches_all, key=lambda x: x.distance)
            self.matches = matches_all[:self.BF_LIMIT]
\
        elif self.matcher_name == 'FLANN':
            # FLANN parameters
            index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=4)
            search_params = dict(checks=self.FLANN_CHECKS)

            # flann matching
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            # matching descriptor vectors using FLANN Matcher
            matches_all = flann.knnMatch(self.des1, self.des2, k=2)

            # Lowe's ratio test to filter matches_img
            self.matches = []
            for m, n in matches_all:
                if m.distance < self.FLANN_RATIO_TH * n.distance:
                    self.matches.append(m)
            
            if DYNAMIC_FLANN_RATIO_TH:
                if len(self.matches) > 6 and self.FLANN_RATIO_TH > 0.3:
                    self.FLANN_RATIO_TH = self.FLANN_RATIO_TH - 0.02                    
                elif len(self.matches) < 6 and self.FLANN_RATIO_TH < 0.8:
                    self.FLANN_RATIO_TH = self.FLANN_RATIO_TH + 0.02                

        else:
            raise ValueError('Invalid matcher name')

        if len(self.matches) > 0:
            # get the points
            self.kpt1_src_pts = np.float32([self.kpt1[m.queryIdx].pt for m in self.matches]).reshape(-1, 1, 2)
            self.kpt2_dst_pts = np.float32([self.kpt2[m.trainIdx].pt for m in self.matches]).reshape(-1, 1, 2)

        # measure time
        end = cv2.getTickCount()
        self.time = (end - start) / cv2.getTickFrequency()
        print('Time: %.2fs' % self.time)
        print('Matches: {}'.format(len(self.matches)))

        return self.matches

    def get_target_coord(self):
        # Кластеризация точек (нахождение кластеров точек находящихся рядом)
        reference_height, reference_width = self._get_image_size(self.img1)
        radius = int(max(reference_width, reference_height) * 1.1)
        
        matchesInCluster = int(len(self.matches) / 4)
        if matchesInCluster < 3:
            matchesInCluster = 3
        print('Matches in cluster: {}'.format(matchesInCluster))
        if self.kpt2_dst_pts is None:
            raise('No mathes kaypoint')

        clusters = self._cluster_points(self.kpt2_dst_pts, radius, matchesInCluster)
        
        #Если есть хотябы один кластер и внутри него есть больше двух точек рядом
        if len(clusters) > 0 and len(clusters[0]) > 0:
            # Нахождение центроидов кластеров (нахождение средней точки внутри кластера)
            centroids = self._compute_centroids(clusters, self.kpt2_dst_pts)
            for centroid in centroids:
                x, y = map(int, centroid.astype(int)[0]) # КООРДИНАТЫ ЦЕЛИ !
                return x, y

    def homography(self, plot=True):
        if len(self.matches) > self.HOMOGRAPHY_MATCH_TH:            
            # compute the homography matrix
            M, mask = cv2.findHomography(self.kpt1_src_pts, self.kpt2_dst_pts, cv2.RANSAC, self.HOMOGRAPHY_RANSAC_TH)
            matchesMask = mask.ravel().tolist()

            h, w = self._get_image_size(self.img1)
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            if plot:
                self.img2 = cv2.polylines(self.img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            return M, dst, matchesMask

        else:
            print("Not enough matches are found - {}/{}".format(len(self.matches), self.HOMOGRAPHY_MATCH_TH))
            return None, None, None

    # Call function saveMatcher
    def plot_matches(self, targetCoordX=0, targetCoordY=0):
        """
        Shows the matches_img by plotting them on the images and saving them in the Results folder
        :param matches_img: input matches_img (optional: if not provided, use self.matches_img)
        """
        # Create a new figure
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.axis('off')
        ax.margins(0)

        # Plot the images
        M, dst, matchesMask = self.homography()
        # draw good matches_img
        self.matches_img = cv2.drawMatches(self.img1, self.kpt1, self.img2, self.kpt2, self.matches,
                                          matchesMask=matchesMask, outImg=None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if targetCoordX != 0 and targetCoordY != 0:
            cv2.circle(self.matches_img, (targetCoordX, targetCoordY), 30, (0,255,0))

        ax.imshow(self.matches_img)
        ax.title.set_text('Matcher: ' + self.matcher_name +
                          ',  Detector: ' + self.detector_name +
                          ',  Descriptor: ' + self.descriptor_name)
        # put the time in the bottom right corner and the number of matches in the bottom left corner
        ax.text(0.99, 0.01, 'Time: %.2fs' % self.time, color='orange',
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=10)
        ax.text(0.01, 0.01, 'Matches: {}'.format(len(self.matches)), color='orange',
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, fontsize=10)

        # save and show the figure
        fig.tight_layout()
        plt.savefig('Results/%s-with-%s-%s.png' % (self.matcher_name, self.detector_name, self.descriptor_name),
                    bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    def _distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def _cluster_points(self, points, eps, min_samples):
        # points - Массив точек
        # eps - Радиус окрестности для определения близости
        # min_samples - Минимальное количество точек для формирования кластера
        clusters = []
        visited = set()
        for i, point in enumerate(points):
            if i in visited:
                continue
            cluster = []
            queue = [i]
            while queue:
                idx = queue.pop(0)
                if idx in visited:
                    continue
                visited.add(idx)
                cluster.append(idx)
                for j, other_point in enumerate(points):
                    if j not in visited and self._distance(points[idx], other_point) < eps:
                        queue.append(j)
            if len(cluster) >= min_samples:
                clusters.append(cluster)
        return clusters

    def _compute_centroids(self, clusters, points):
        centroids = []
        for cluster in clusters:
            cluster_points = np.array([points[i] for i in cluster])
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        return centroids

    def _processColor(self, img):
        #return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)
        #return LAB_Color_Space(img)

    def _get_image_size(self, image):
        """
        Возвращает размер изображения в виде строки с указанием высоты, ширины и количества каналов.
        Если изображение одноканальное, каналы не указываются.
        """
        if len(image.shape) == 2:
            height, width = image.shape
            return height, width            
        elif len(image.shape) == 3:
            height, width, channels = image.shape
            return height, width
        else:
            return None