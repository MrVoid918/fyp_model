import numpy as np

from torch import Tensor

from typing import Union, List
from numpy.typing import NDArray

#! Design Pattern to use: Adapter
class CommonOutputFormat:
    def __init__(self) -> None:
        pass

class MOTConverter:

    def __call__(self, 
                 bbox, 
                 frame: int = -1):
        '''
        Converts Detection Outputs to MOT format
        [frame, id, bb_left, bb_right, bb_width, bb_height, conf, x_3d, y_3d, z_3d]

        Current Implemented:
            1. PicoDet
            2. Yolov7 [([bbox], scores, classid)] Tensor [[bbox, scores, classid]]]
            3. 
        '''
        mot = None
        if isinstance(bbox, List):
            # Yolov7
            bbox = bbox[0]
            x = np.array(bbox)

            coordinates = np.vstack(x[:, 0])
            confidence_scores = x[:, 1]
            
            mot = np.concatenate((coordinates, confidence_scores[:, np.newaxis]), axis=1)
            #? Pad end and front with -1, as x_3d and y_3d
            mot = np.pad(mot, ((0, 0), (1, 3)), mode='constant', constant_values=-1)
        elif isinstance(bbox, Tensor):
            bbox = bbox.cpu().detach().numpy()
            # [[bbox, scores, classid]]
            bbox[:, 5] = -1
            mot = np.pad(bbox, ((0, 0), (1, 3)), mode='constant', constant_values=-1)

        assert mot is not None, "Invalid input to MOTConverter"
        mot = np.pad(mot, ((0, 0), (1, 0)), mode='constant', constant_values=frame)
        return mot

    def sort_to_mot(self, ori_data: NDArray, frame: int):
        """
        Input: [[x1,y1,x2,y2,id],[x1,y1,x2,y2,id], ...]
        Output: [[frame, id, x1, y1, w, h, 1, -1, -1, -1], ...]
        """
        data = ori_data.copy()
        data[:, 2:4] -= data[:, 0:2] # [x1, y1, x2, y2] -> [x1, y1, w, h]
        data = np.roll(data, 1, axis=1) # [x1, y1, w, h, id] -> [id, x1, y1, w, h]
        data = np.pad(data, ((0, 0), (1, 0)), mode='constant', constant_values=frame)    # [frame, id, x1, y1, w, h]
        data = np.pad(data, ((0, 0), (0, 1)), mode='constant', constant_values=1)    # [frame, id, x1, y1, w, h, 1]
        data = np.pad(data, ((0, 0), (0, 3)), mode='constant', constant_values=-1)    # [frame, id, x1, y1, w, h, 1, -1, -1, -1]

        return data