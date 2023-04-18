from mmdet.apis import init_detector, inference_detector
import mmcv

class YoloX:

    def __init__(self, config_file, checkpoint_file, device='cuda:0'):
        config_path = f"mmdetection/configs/yolox/{config_file}"
        self.model = init_detector(config_path, checkpoint_file, device=device)

    def predict(self, img):
        img = mmcv.imread(img)
        result = inference_detector(self.model, img)
        return result

    def __call__(self, img):
        return self.predict(img)