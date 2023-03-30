from models.experimental import attempt_load
from pathlib import Path
import torch
import torch.nn as nn


from utils.torch_utils import TracedModel

class Yolov7(nn.Module):

    def __init__(self, 
                 weights_path: Path, 
                 device: torch.device, 
                 img_size: int) -> None:
        self.device = device
        model = attempt_load(weights=weights_path, map_location=device)  # load FP32 model
        self.model = TracedModel(model, device, img_size)
        self.model.half()
        

    def forward(self):
        pass

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # uint8 to fp16/32, as we currently work with GPU, we default this
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        