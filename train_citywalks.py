from jaad.jaad import parse_sgnet_args
from models.traj_pred import TrajPred, Decoder, TrajConcat, Loss, TrajPredGRU, DecoderGRU
from dataloaders.data_utils import build_data_loader, bbox_normalize, bbox_denormalize
from dataloaders.citywalks import CityWalksDataset
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import traceback
import numpy as np
from numpy.typing import NDArray

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchreid.utils.avgmeter import AverageMeter
from train import evaluate

def convert_cxcywh_to_xywh(bbox: NDArray | torch.Tensor):
    converted_bbox = bbox.copy()
    converted_bbox[..., 0] = bbox[..., 0] - bbox[..., 2] / 2
    converted_bbox[..., 1] = bbox[..., 1] - bbox[..., 3] / 2
    return converted_bbox


if __name__ == '__main__':
    args = parse_sgnet_args()
    train_dir = [
        Path("d:/Downloads/ground_truth") / f"train_mask-rcnn_fold_{i}.csv" for i in range(1, 2)
    ]
    val_dir = [
        Path("d:/Downloads/ground_truth") / f"val_mask-rcnn_fold_{i}.csv" for i in range(1, 2)
    ]

    logger.remove()
    logger.add("logs/{time}.log", level="INFO", rotation="1 day",)

    if args.gpu:
        device = torch.device("cuda", int(args.gpu))
    else:
        device = torch.device("cpu")

    logger.info("Loading Dataset")
    train_data = np.load("tracks_all.npy").astype(np.float32)
    val_data = np.load("tracks_rev.npy").astype(np.float32)
    input_x = convert_cxcywh_to_xywh(train_data[:, :args.enc_steps, :])
    val_input_x = convert_cxcywh_to_xywh(val_data[:, :args.enc_steps, :])
    if args.velocity:
        velocity = np.diff(input_x, n = 1, axis = 1)
        velocity = np.pad(velocity, ((0, 0), (1, 0), (0, 0)), constant_values = 0)
        input_x = np.concatenate((input_x, velocity), axis = -1)

        val_velocity = np.diff(val_input_x, n = 1, axis = 1)
        val_velocity = np.pad(val_velocity, ((0, 0), (1, 0), (0, 0)), constant_values = 0)
        val_input_x = np.concatenate((val_input_x, val_velocity), axis = -1)
    target_y = convert_cxcywh_to_xywh(train_data[:, args.enc_steps:, :])
    val_target_y = convert_cxcywh_to_xywh(val_data[:, args.enc_steps:, :])


    # train_dataset = CityWalksDataset(train_dir)
    train_dataset = TensorDataset(torch.from_numpy(input_x), torch.from_numpy(target_y))
    val_dataset = TensorDataset(torch.from_numpy(val_input_x), 
                                torch.from_numpy(val_target_y))
    # val_dataset = CityWalksDataset(val_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    logger.success("Successfully Loaded Dataset")

    logger.info("Loading Models")
    traj_pred = TrajPred(
            input_size=args.input_dim,
            hidden_size=args.hidden_size,
            output_size=256,
        ).to(device)

    decoder = Decoder(
        input_size=256,
        hidden_size=args.hidden_size,
        output_size=4,
        frames_future=args.dec_steps, 
    ).to(device)

    traj_concat = TrajConcat(args=args).to(device)

    enc_optimizer = Adam(traj_pred.parameters(), lr=args.lr)
    dec_optimizer = Adam(decoder.parameters(), lr=args.lr)

    objective = Loss(args = args)
    logger.success("Successfully Loaded Models")

    writer = SummaryWriter("logs/citywalks")

    val_mse_meter = AverageMeter()
    val_cmse_meter = AverageMeter()
    val_cfmse_meter = AverageMeter()

    test_mse_meter = AverageMeter()
    test_cmse_meter = AverageMeter()
    test_cfmse_meter = AverageMeter()

    # ----------------- Training ----------------- #
    try:
        for epoch in tqdm(range(1, args.epochs + 1)):
            logger.info(f"Epoch {epoch} of {args.epochs}")
            traj_pred.train()
            decoder.train()
            train_loss = 0
            for i, (input_x, target_y) in tqdm(enumerate(train_loader, start = 1), total=len(train_loader), leave=False):
                input_x = input_x.to(device)
                target_y = target_y.to(device)

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                input_vec = input_x.flip(dims = [1])

                enc_output, dec_output, enc_h = traj_pred(input_x)
                dec_result = decoder(enc_output, enc_h)
                end_result = traj_concat(dec_result, input_vec)

                loss = objective(dec_output, input_vec, end_result, target_y, )

                train_loss += loss

                loss.backward()
                
                dec_optimizer.step()
                enc_optimizer.step()

            writer.add_scalar('Loss/train', train_loss, epoch)
            logger.info(f"Train Loss: {train_loss}")

            # ---------- Training Evaluation ---------- #
            with torch.no_grad():
                val_loss = 0
                traj_pred.eval()
                decoder.eval()
                for i, (input_x, target_y) in tqdm(enumerate(val_loader, start = 1), total=len(val_loader), leave=False):
                    input_x = input_x.to(device)
                    target_y = target_y.to(device)

                    enc_optimizer.zero_grad()
                    dec_optimizer.zero_grad()

                    input_vec = input_x.flip(dims = [1])

                    enc_output, dec_output, enc_h = traj_pred(input_x)
                    dec_result = decoder(enc_output, enc_h)
                    end_result = traj_concat(dec_result, input_vec)

                    loss = objective(dec_output, input_vec, end_result, target_y, )

                    val_loss += loss

                    val_mse_, val_cmse_, val_cfmse_ = evaluate(args, end_result, target_y)

                    val_mse_meter.update(val_mse_)
                    val_cmse_meter.update(val_cmse_)
                    val_cfmse_meter.update(val_cfmse_)

            writer.add_scalar('Loss/val', val_loss, epoch)
            logger.info(f"Validation Loss: {val_loss}")
            logger.info(f"MSE: {val_mse_meter.avg}")
            logger.info(f"CMSE: {val_cmse_meter.avg}")
            logger.info(f"CFMSE: {val_cfmse_meter.avg}")

    except:
        traceback.print_exc()

    finally:
        torch.save({'enc': traj_pred.state_dict(), 
                    'dec': decoder.state_dict()}, 
                    Path(args.save_weights))