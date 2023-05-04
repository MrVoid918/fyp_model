from abc import ABC, abstractmethod
import cv2
import numpy as np

class CMC:
    def __init__(self, strategy: str, scale = 0.1, ) -> None:
        if strategy == "ECC":
            self.strategy = ECC(scale = scale)
        elif strategy == "ORB":
            self.strategy = ORBRANSAC()

    def __call__(self, img1, img2):
        return self.strategy.generate_homography(img1, img2)

class CMCStrategy(ABC):
    """
    Base Class to Perform Camera Motion Compensation
    Input: 2 images
    Output: Homography Matrix
    """

    @abstractmethod
    def generate_homography(self, img1, img2):
        pass

class ECC(CMCStrategy):
    """
    ECC Camera Motion Compensation
    """

    def __init__(self, scale) -> None:
        super().__init__()
        self.warp_mode = cv2.MOTION_EUCLIDEAN
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        self.num_of_iterations = 1000
        self.termination_eps = 1e-3
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.num_of_iterations, self.termination_eps)
        self.scale = scale

    def generate_homography(self, img1, img2):
        if img1.shape[-1] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.shape[-1] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        if self.scale is not None:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale != 1:
                    src_r = cv2.resize(img1, (0, 0), fx = self.scale, fy = self.scale,interpolation =  cv2.INTER_LINEAR)
                    dst_r = cv2.resize(img2, (0, 0), fx = self.scale, fy = self.scale,interpolation =  cv2.INTER_LINEAR)
                    scale = [self.scale, self.scale]
                else:
                    src_r, dst_r = img1, img2
                    scale = None
        else:
            src_r, dst_r = img1, img2

        try:
            (cc, warp_matrix) = cv2.findTransformECC(src_r, 
                                                     dst_r, 
                                                     self.warp_matrix, 
                                                     self.warp_mode, 
                                                     self.criteria, None, 1)
        except cv2.error as e:
            print('ecc transform failed')
            return None, None
        
        if self.scale is not None:
            warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
            warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

        return warp_matrix

        _, self.warp_matrix = cv2.findTransformECC(img1, img2, self.warp_matrix, self.warp_mode, self.criteria, inputMask=None, gaussFiltSize=1)
        return self.warp_matrix

class ORBRANSAC(CMCStrategy):
    """
    ORB + RANSAC Camera Motion Compensation
    """

    def __init__(self) -> None:
        super().__init__()
        self.orb = cv2.ORB_create()

        FLANN_INDEX_LSH    = 6
        self.index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        self.search_params = {"checks": 50}
        self.min_matches = 50

        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def generate_homography(self, img1, img2):
        if img1.shape[-1] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.shape[-1] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kpts1, descs1 = self.orb.detectAndCompute(img1, None)
        kpts2, descs2 = self.orb.detectAndCompute(img2, None)

        matches = self.flann.knnMatch(descs1, descs2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        homo_mat = None

        if len(good_matches) > self.min_matches:
            src_points = np.float32([kpts1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_points = np.float32([kpts2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            homo_mat, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        return homo_mat
