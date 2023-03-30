import numpy as np

from pathlib import Path
from fyp_model.models.traj_pred import TrajPred, Decoder, TrajConcat
import torch
import torch.nn as nn

from time import time

from loguru import logger

class PredictTrajectory(nn.Module):

    def __init__(self, 
                 file: str, 
                 output_file: str,
                 model_path: str, 
                 img_width: int, 
                 img_height: int, 
                 past_frames = 45, 
                 future_frames = 60, 
                 input_size = 4, 
                 enc_hidden_size = 512, 
                 enc_output_size = 256, 
                 normalize = False, 
                 velocity = False, 
                 ) -> None:
        super().__init__()
        try:
            self.file = Path(file).resolve()
            assert self.file.exists(), "File does not exist"
        except FileNotFoundError:
            logger.error("File not found")
            raise FileNotFoundError
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.img_width = img_width
        self.img_height = img_height
        self.normalize = normalize
        self.velocity = velocity

        self.output_file = Path(output_file).resolve()
        self.output_file.touch(exist_ok=True)

        self.model_path = Path(model_path).resolve()
        assert self.model_path.exists(), "Model Weights does not exist"

        self.enc = TrajPred(
            input_size=input_size,
            hidden_size=enc_hidden_size,
            output_size=enc_output_size,
        )
        self.dec = Decoder(
            input_size=enc_output_size,
            hidden_size=enc_hidden_size,
            output_size=4,
            frames_future=future_frames,
        )

        self.traj_cat = TrajConcat()

        checkpoint = torch.load(self.model_path)

        self.enc.load_state_dict(checkpoint['enc'])
        self.dec.load_state_dict(checkpoint['dec'])

        self.enc.eval()
        self.dec.eval()
        self.traj_cat.eval()

        self._save_format = fmt = '%d', '%d', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%d', '%d', '%d', '%d'

    def preprocess(self, data):
        # Normalize data
        # xyxy to cxcywh
        # xywh to cxcywh
        bbox = data[:, 2:6]
        # bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
        bbox[..., [0, 1]] += bbox[..., [2, 3]]/2
        # W, H  = all_resolutions[i][0]
        if self.velocity:
            velocity = np.diff(bbox, n=1, axis=0)
            velocity = np.concatenate((np.zeros((1, 4)), velocity, ), axis=0)
            bbox = np.concatenate((bbox, velocity), axis=1)

        if self.normalize:
            min_bbox = np.array([0, 0, 0, 0])
            max_bbox = np.array([self.img_width, self.img_height, self.img_width, self.img_height])
            _min = np.array(min_bbox)[None, :]
            _max = np.array(max_bbox)[None, :]
            bbox = (bbox - _min) / (_max - _min)
        return bbox

    def load(self):
        self.file = np.loadtxt(self.file, delimiter=',')

    def predict(self, person_id: int, frame_id: int):
        # Get most recent data written to text file
        # data => xywh
        # processed_data => cxcywh

        t1 = time()
        self.load()
        data = self.file[np.where((self.file[:, 1] == person_id) & \
                         (self.file[:, 0] <= frame_id) & \
                         (self.file[:, 0] >= np.max(frame_id - self.past_frames, 0)))]

        if not data.size:
            logger.error("No data found, Returning zeros")
            data =  np.zeros((self.future_frames, 4))
            data[:, 1] = np.arange(frame_id, frame_id + self.future_frames)
            return data

        processed_data = self.preprocess(data)
        torch_data = torch.flipud(torch.from_numpy(processed_data)).float().cuda()
        
        if torch_data.ndim == 2:
            torch_data = torch_data.unsqueeze(0)

        with torch.no_grad():
            logger.info(f"Predicting for {person_id} at {frame_id}")
            enc_output, dec_output, enc_h = self.enc(torch_data)
            pred = self.dec(enc_output, enc_h)
            pred = self.traj_cat(pred, torch_data)

        pred = pred.cpu().numpy()

        bbox = data[-1, 2:6]
        result = PredictTrajectory.process_cxcywh_to_xywh(pred)
        # We have already added the velocity in the TrajConcat layer
        # result = bbox + processed_preds[0, ::-1, :]
        # result = bbox + processed_preds[0, ...]

        result = self._format_MOT_output(p_id=person_id, frame=frame_id, bbox=result)

        np.savetxt(self.output_file, result, delimiter=',', fmt=self._save_format)

        t2 = time()

        logger.info(f"Predicted for person id: {person_id} at frame id: {frame_id} to {frame_id + self.future_frames}")
        logger.info(f"Time required {t2 - t1}ms")

        return result

    @staticmethod
    def bbox_normalize(bbox,W=1280,H=640):
        '''
        normalize bbox value to [0,1]
        :Params:
            bbox: [cx, cy, w, h] with size (times, 4), value from [0, max(W, H)]
        :Return:
            bbox: [cx, cy, w, h] with size (times, 4), value from [0, 1]
        '''
        new_bbox = bbox
        new_bbox[..., 0] *= W
        new_bbox[..., 1] *= H
        new_bbox[..., 2] *= W
        new_bbox[..., 3] *= H
        
        return new_bbox

    @staticmethod    
    def bbox_denormalize(self, bbox, W=1280,H=640):
        '''
        denormalize bbox value from [0,1]
        :Params:
            bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
        :Return:
            bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
        '''
        new_bbox = bbox
        new_bbox[..., 0] *= self.image_width
        new_bbox[..., 1] *= self.image_height
        new_bbox[..., 2] *= self.image_width
        new_bbox[..., 3] *= self.image_height
        
        return new_bbox

    @staticmethod
    def process_cxcywh_to_xywh(bbox):
        bbox[..., :2] -= bbox[..., 2:] / 2.
        return bbox

    def _format_MOT_output(self, p_id, frame, bbox, ):
        """
        Format the output in MOT format
        The format is as follows:
        frame, id, x, y, w, h, conf, -1, -1, -1

        Parameters: 
        @params: p_id: Pedestrian ID
        @frame: Starting Frame
        @bbox: Predicted Bounding Boxes Coordinates

        Returns:
        @result: MOT formatted output
        """
        frame_id = frame + np.arange(1, self.future_frames + 1).reshape(-1, 1)

        if bbox.ndim == 3:
            bbox = np.squeeze(bbox, axis=0)
        assert bbox.shape == (self.future_frames, 4), f"Bbox shape is {bbox.shape} instead of {(self.future_frames, 4)}"

        result = np.pad(bbox, ((0, 0), (1, 1)), 'constant', constant_values=(p_id, 1))
        assert result.shape == (self.future_frames, 6), f"Result shape is {result.shape} instead of {(self.future_frames, 6)}"

        result = np.pad(result, ((0, 0), (0, 3)), 'constant', constant_values=-1)
        result = np.hstack((frame_id, result, ))

        assert result.shape == (self.future_frames, 10), f"Result shape is {result.shape} instead of {(self.future_frames, 10)}"
        return result