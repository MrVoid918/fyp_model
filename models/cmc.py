from abc import ABC, abstractmethod
import cv2
import numpy as np

class CMC:
    def __init__(self, strategy: str) -> None:
        if strategy == "ECC":
            self.strategy = ECC()

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

    def __init__(self) -> None:
        super().__init__()
        self.warp_mode = cv2.MOTION_HOMOGRAPHY
        self.warp_matrix = np.eye(3, 3, dtype=np.float32)
        self.num_of_iterations = 500
        self.termination_eps = 1e-3
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.num_of_iterations, self.termination_eps)

    def generate_homography(self, img1, img2):
        if img1.shape[-1] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if img2.shape[-1] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        _, self.warp_matrix = cv2.findTransformECC(img1, img2, self.warp_matrix, self.warp_mode, self.criteria, inputMask=None, gaussFiltSize=1)
        return self.warp_matrix

