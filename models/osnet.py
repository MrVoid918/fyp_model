from torchreid.utils import FeatureExtractor

from pathlib import Path
from numpy.typing import NDArray
import torch
from typing import List

class OSNet:

    def __init__(self, path: str) -> None:

        self.extractor = FeatureExtractor(
            model_name='osnet_x0_25',
            model_path= Path(path).resolve(),
            device='cuda'
        )

    def extract_features(self, imgs: List[NDArray], normalize = True):
        features = self.extractor(imgs)
        return features / torch.linalg.norm(features, dim=1, keepdim=True)

    def __call__(self, imgs: List[NDArray]):
        return self.extract_features(imgs)