from pie.pie import parse_sgnet_args
from dataloaders.data_utils import build_data_loader
from models.traj_pred import TrajPred, Decoder, TrajConcat, Loss
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.io as vision_io
import torchvision.utils as vision_utils
import torchvision.ops as vision_ops

from loguru import logger
import itertools

import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, OneCycleLR

if __name__ == '__main__':
    pie_args = parse_sgnet_args()
    pie_dataloader = build_data_loader(pie_args)

    jaad_dataloader = build_data_loader(pie_args)
    val_jaad_dataloader = build_data_loader(pie_args, 'val')
    test_jaad_dataloader = build_data_loader(pie_args, 'test')

    traj_pred = TrajPred(
            input_size=pie_args.input_dim,
            hidden_size=pie_args.hidden_size,
            output_size=256,
        ).to('cuda:0')

    decoder = Decoder(
        input_size=256,
        hidden_size=pie_args.hidden_size,
        output_size=4,
        frames_future=pie_args.dec_steps, 
    ).to('cuda:0')

    traj_concat = TrajConcat(args=pie_args).to('cuda:0')

    enc_optimizer = Adam(traj_pred.parameters(), lr=pie_args.lr)
    dec_optimizer = Adam(decoder.parameters(), lr=pie_args.lr)

    
    enc_scheduler = CosineAnnealingLR(enc_optimizer, T_max=pie_args.epochs)
    dec_scheduler = CosineAnnealingLR(dec_optimizer, T_max=pie_args.epochs)

    # enc_scheduler = OneCycleLR(enc_optimizer, 
    #                            max_lr=jaad_args.lr * 10, 
    #                            epochs=jaad_args.epochs, 
    #                            steps_per_epoch=len(jaad_dataloader))

    # dec_scheduler = OneCycleLR(dec_optimizer, 
    #                            max_lr=jaad_args.lr * 10, 
    #                            epochs=jaad_args.epochs, 
    #                            steps_per_epoch=len(jaad_dataloader))

    objective = Loss(args = pie_args)

    # enc_scheduler = StepLR(enc_optimizer, step_size=5, gamma=0.5)
    # dec_scheduler = StepLR(dec_optimizer, step_size=5, gamma=0.5)

    writer = SummaryWriter()

    '''
    path_anno = Path('inference/seq/gt.txt').resolve()
    assert path_anno.exists(), f"{path_anno} does not exist"
    file = np.loadtxt(path_anno, delimiter = ',')
    bboxes = torch.from_numpy(file[np.where((file[:, 1] == 1) & \
                         (file[:, 0] <= jaad_args.enc_steps + jaad_args.dec_steps) & \
                         (file[:, 0] > jaad_args.enc_steps))][:, 2:6].astype(int))

    enc_bboxes = torch.from_numpy(file[np.where((file[:, 1] == 1) & \
                         (file[:, 0] <= jaad_args.enc_steps))][:, 2:6].astype(int))

    bboxes = vision_ops.box_convert(bboxes, in_fmt = 'xywh', out_fmt = 'xyxy')

    try:
        for epoch in range(1, jaad_args.epochs + 1):
            logger.info(f"Epoch: {epoch}")
            traj_pred.train()
            decoder.train()
            train_loss = 0
            for i, data in enumerate(jaad_dataloader):
                
                input_vec = data['input_x'].flip(dims = [1]).to('cuda:0')
                enc_output, dec_output, enc_h = traj_pred(input_vec)

                if jaad_args.velocity:
                    dec_result = decoder(enc_output, enc_h)
                    end_result = traj_concat(dec_result, input_vec)
                else:
                    end_result = decoder(enc_output, enc_h)

                ground_truth = data['target_y'][:, -1, :, :4].to('cuda:0')

                assert end_result.shape == ground_truth.shape, f"{end_result.shape}, {ground_truth.shape}"

                loss = objective(dec_output, input_vec, end_result, ground_truth, )

                train_loss += loss

                logger.info(f"Batch: {i + 1} Loss: {loss}")

                enc_optimizer.zero_grad()
                dec_optimizer.zero_grad()

                loss.backward()

                dec_optimizer.step()
                enc_optimizer.step()

                dec_scheduler.step()
                enc_scheduler.step()

            writer.add_scalar('Loss/train', train_loss, epoch)

            with torch.no_grad():
                val_loss = 0
                traj_pred.eval()
                decoder.eval()
                for i, data in enumerate(val_jaad_dataloader):

                    input_vec = data['input_x'].flip(dims = [1]).to('cuda:0')
                    enc_output, dec_output, enc_h = traj_pred(input_vec)

                    if jaad_args.velocity:
                        dec_result = decoder(enc_output, enc_h)
                        end_result = traj_concat(dec_result, input_vec)
                    else:
                        end_result = decoder(enc_output, enc_h)

                    ground_truth = data['target_y'][:, -1, :, :4].to('cuda:0')

                    assert end_result.shape == ground_truth.shape, f"{end_result.shape}, {ground_truth.shape}"

                    loss = objective(dec_output, input_vec, end_result, ground_truth, )

                    val_loss += loss

                logger.info(f"Validation Loss: {val_loss}")

            with torch.no_grad():
                test_loss = 0
                traj_pred.eval()
                decoder.eval()
                for i, data in enumerate(test_jaad_dataloader):

                    input_vec = data['input_x'].flip(dims = [1]).to('cuda:0')
                    enc_output, dec_output, enc_h = traj_pred(input_vec)
                    
                    if jaad_args.velocity:
                        dec_result = decoder(enc_output, enc_h)
                        end_result = traj_concat(dec_result, input_vec)
                    else:
                        end_result = decoder(enc_output, enc_h)

                    ground_truth = data['target_y'][:, -1, :, :4].to('cuda:0')

                    assert end_result.shape == ground_truth.shape, f"{end_result.shape}, {ground_truth.shape}"

                    loss = objective(dec_output, input_vec, end_result, ground_truth, )

                    test_loss += loss

                logger.info(f"Test Loss: {test_loss}")

            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)

            if jaad_args.show_result and epoch % jaad_args.show_interval == 0:
                # Use example from MOT17 to show the result
                path = Path('inference/images/img1/').resolve()
                assert path.exists(), f"{path} does not exist"

                enc_bbox = vision_ops.box_convert(enc_bboxes, in_fmt = 'xywh', out_fmt = 'cxcywh')
                if jaad_args.velocity:
                    velocity = torch.diff(enc_bbox, n=1, axis=0)
                    velocity = torch.concatenate((torch.zeros((1, 4)), velocity, ), dim=0)
                    enc_bbox = torch.concatenate((enc_bbox, velocity), dim=1)

                input_data = torch.flipud(enc_bbox).float().cuda()

                if input_data.ndim == 2:
                    input_data = input_data.unsqueeze(0)

                with torch.no_grad():
                    enc_output, dec_output, enc_h = traj_pred(input_data)
                    pred = decoder(enc_output, enc_h)
                    pred = traj_concat(pred, input_data)
                    pred = vision_ops.box_convert(pred, in_fmt = 'cxcywh', out_fmt = 'xyxy')

                im_files = []
                # We start enumerate from future frames
                for frame_id, im_path in enumerate(sorted(list(path.iterdir()))[jaad_args.enc_steps+1:jaad_args.enc_steps + jaad_args.dec_steps + 1], 
                                        start = 1):
                    im = vision_io.read_image(str(im_path))
                    bbox = bboxes[[frame_id - 1], :]
                    pred_bbox = pred[0, [frame_id - 1], :]
                    if pred_bbox.ndim == 1:
                        pred_bbox = pred_bbox.unsqueeze(0)
                    im = vision_utils.draw_bounding_boxes(im, bbox, colors = (255, 0, 0), width = 3)
                    im = vision_utils.draw_bounding_boxes(im, pred_bbox, colors = (0, 0, 255), width = 3)
                    im_files.append(im)

                vid = torch.stack(im_files, dim = 0)

                if vid.ndim == 4:
                    vid = vid.unsqueeze(0)

                writer.add_video('sampled_trajectory', vid, epoch, fps = 10)

            writer.flush()

    except KeyboardInterrupt:
        pass

    finally:
        torch.save({'enc': traj_pred.state_dict(), 
                    'dec': decoder.state_dict()}, 
                    Path(jaad_args.save_weights))

    # target_y is the change of values, consider then as the velocity
    # torch.Size([128, 15, 4])
    # torch.Size([128, 15, 45, 4])
    # Using past 15 to predict 45 future frames?
    '''