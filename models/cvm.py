import torch
from pathlib import Path
import numpy as np


from loguru import logger

class CVM:

    def __init__(self, trk_file: str, past_frames=30, future_frames = 40, sample = False, xy = False):
        """
        @trk_file: Path string to the tracking file
        @past_frames: Number of past frames to consider for prediction
        @future_frames: Number of future frames to predict
        @sample: Sample from Gaussian Distribution
        @xy: Predict with xy only, omit bbox width and height
        """
        self.trk_file = Path().cwd() / 'output' / trk_file
        assert self.trk_file.exists(), f"File {self.trk_file} does not exist"
        self.past_frames = past_frames
        self.future_frames = future_frames

        self.sample = sample
        self.xy = xy

        self._save_format = fmt = '%d', '%d', '%1.2f', '%1.2f', '%1.2f', '%1.2f', '%d', '%d', '%d', '%d'

    def __call__(self, p_id: int, frame: int):
        """This function performs prediction and saves the output in a file
        @p_id: Pedestrian ID
        @frame: Starting Frame
        """
        # MOT format
        # frame, id, x, y, w, h, conf, -1, -1, -1
        logger.info("Performing Pedestrian Trajectory Prediction")
        self.file = self._load()
        data  = self.file[np.where((self.file[:, 1] == p_id) & \
                           (self.file[:, 0] <= frame) & \
                           (self.file[:, 0] >= np.max(frame - self.past_frames, 0)))]

        assert len(data) > 0, f"Person {p_id} not found in frame {frame}"

        bbox_data = data[:, 2:4] if self.xy else data[:, 2:6]
        vel = np.diff(bbox_data, n=1, axis=0)

        mean_vel = vel.mean(axis=0)
        if self.sample:
            # Sample Gaussian Distribution
            # TODO: Perform CVM only on Center cxcy where it is abnormal to perform prediction and increment
            # TODO: on bounding box coordinates
            sigma = np.diag(np.full(bbox_data.shape[-1], 1))
        else:
            sigma = np.zeros((bbox_data.shape[-1], bbox_data.shape[-1]))

        future_vel = np.random.default_rng().multivariate_normal(mean_vel, sigma, (self.future_frames, ))

        future_increments = np.cumsum(future_vel, axis=0)
        future_positions = bbox_data[-1, :] + future_increments

        result = self._format_MOT_output(p_id, frame, future_positions, data)

        np.savetxt("pred.txt", result, delimiter=",", fmt=self._save_format)

        logger.info("Completed Pedestrian Trajectory Prediction")

    def _load(self):
        return np.loadtxt(self.trk_file, delimiter=',')

    def _format_MOT_output(self, p_id, frame, bbox, data):
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
        if self.xy:
            bbox = np.hstack((bbox, np.tile(data[-1, 4:6], (self.future_frames, 1))))

        assert bbox.shape == (self.future_frames, 4), f"Bbox shape is {bbox.shape} instead of {(self.future_frames, 4)}"

        result = np.pad(bbox, ((0, 0), (1, 1)), 'constant', constant_values=(p_id, 1))
        assert result.shape == (self.future_frames, 6), f"Result shape is {result.shape} instead of {(self.future_frames, 6)}"

        result = np.pad(result, ((0, 0), (0, 3)), 'constant', constant_values=-1)
        result = np.hstack((frame_id, result, ))

        assert result.shape == (self.future_frames, 10), f"Result shape is {result.shape} instead of {(self.future_frames, 10)}"
        return result