from torchreid.utils import FeatureExtractor

from pathlib import Path
from numpy.typing import NDArray
from typing import List

class OSNet:

    def __init__(self) -> None:

        self.extractor = FeatureExtractor(
            model_name='osnet_x0_25',
            model_path= Path('weights/osnet_x0_25_imagenet.pth').resolve(),
            device='cuda'
        )

    def extract_features(self, imgs: List[NDArray]):
        return self.extractor(imgs)

    def __call__(self, imgs: List[NDArray]):
        return self.extract_features(imgs)