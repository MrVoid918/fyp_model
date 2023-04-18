from jaad.jaad import parse_sgnet_args
from dataloaders.data_utils import build_data_loader
from models.cvm import CVM
import numpy as np

from numpy.typing import NDArray
from typing import Tuple

def evaluate(args, bbox_gt: NDArray, bbox_pred: NDArray) -> \
    Tuple[NDArray, NDArray, NDArray]:
    """
    This function evaluates the performance of the bounding box predictor
    @param bbox_gt: Ground truth bounding boxes => [N, K, 4]
    @param bbox_pred: Predicted bounding boxes => [N, K, 4]
    @return: The MSE between upper left coords and MSE between center coords
    """
    if args.bbox_type == 'cxcywh':
        CMSE = np.linalg.norm(bbox_gt[..., :2] - bbox_pred[..., :2])
        MSE = np.linalg.norm((bbox_gt[..., 2:] + bbox_gt[..., 2:] / 2) - \
            (bbox_pred[..., 2:] + bbox_pred[..., 2:] / 2))
        CFMSE = np.linalg.norm(bbox_gt[-1, ..., :2] - bbox_pred[-1, ..., :2])
        return CMSE, MSE, CFMSE

def eval():
    jaad_args = parse_sgnet_args()
    test_jaad_dataloader = build_data_loader(jaad_args, 'test')

    model = CVM(past_frames=60, future_frames=30, sample=True, xy=True)

    total_CMSE = 0
    total_MSE = 0
    total_CFMSE = 0

    total_samples = 0

    for i, data in enumerate(test_jaad_dataloader):
        # Input data: N x K x 8
        # N: Batch Number
        # K: Number of frames
        # 8: Bounding Box Coordinates, Expected in cx, cy, w, h and it's derivatives format

        input_x = data['input_x'].numpy()
        target_y = data['target_y'][:, -1, :, :].numpy()
        future_increments = model.call_batch(input_x)
        preds = input_x[..., [-1], :4] + future_increments

        CMSE, MSE, CFMSE = evaluate(jaad_args, target_y[..., :4], preds)

        total_CMSE += CMSE
        total_MSE += MSE
        total_CFMSE += CFMSE

    avg_CMSE = total_CMSE / len(test_jaad_dataloader)
    avg_MSE = total_MSE / len(test_jaad_dataloader)
    avg_CFMSE = total_CFMSE / len(test_jaad_dataloader)

    print(f"Average CMSE: {avg_CMSE}, Average MSE: {avg_MSE}, Average CFMSE: {avg_CFMSE}")
    
if __name__ == "__main__":
    eval()