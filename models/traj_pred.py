import torch
import torch.nn as nn

from torchvision.ops import distance_box_iou_loss, box_convert


class TrajPred(nn.Module):
    '''
    Encoder model for trajectory prediction
    Based on the paper: https://arxiv.org/pdf/2011.04943.pdf
    :param input_size: input size of the model = 8
    :param hidden_size: hidden size of the model = 512
    :param output_size: output size of the model = 256
    :param num_layers: number of layers in the model = 1

    Input would be of shape (batch_size, seq_len, input_size)

    From Paper: input = (k, 8) where k is the number of frames in the trajectory
    Output of LSTM will be fed to a fully connected layer with ReLU activation
    Output of the fully connected layer will be of shape (batch_size, 256)

    Output of the FC layer is fed to another LSTM layer which acts as the decoder
    Run k time to reproduce k hidden states
    '''

    def __init__(self, 
                 input_size = 8, 
                 hidden_size = 512, 
                 output_size = 256, 
                 num_layers=1, 
                 ):
        super(TrajPred, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enc_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, output_size), 
            nn.ReLU(),
        )

        self.dec_lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True, )
        self.dec_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        k = x.shape[1]

        # Generate a final hidden state vector
        _ , (enc_h, _) = self.enc_lstm(x)
        init_cell_state = torch.zeros_like(enc_h)
        # enc_h => [D*num_layers, batch_size, hidden_size]
        # we want => [batch_size, ...]
        
        #enc_h_proped = enc_h.squeeze(dim = 1)
        enc_h_proped = enc_h.transpose(0, 1)
        enc_output = self.encoder(enc_h_proped)

        # dec_output = None
        _, (dec_h, cell_state) = self.dec_lstm(enc_output, (enc_h, init_cell_state))
        final_h = dec_h.transpose(0, 1)
        for i in range(k - 1):
            # if dec_output is None:
                # dec_output = self.dec_lstm(enc_output.unsqueeze(1))[0]
                # _ , (dec_h, _) = self.dec_lstm(enc_output)
                # dec_output = dec_h.transpose(0, 1)
                # dec_output = self.dec_lstm(enc_output)[0]
            # else:
                # dec_output = torch.cat((dec_output, self.dec_lstm(enc_output.unsqueeze(1))[0]), dim=1)
            # We reuse previous states from previous iterations of the decoder
            _ , (dec_h, cell_state) = self.dec_lstm(enc_output, (dec_h, cell_state))
                # dec_output = torch.cat((dec_output, dec_h.transpose(0, 1)), dim=1)
            final_h = torch.cat((final_h, dec_h.transpose(0, 1)), dim=1)

        final_h = self.dec_fc(final_h)

        return enc_output, final_h, enc_h

class Decoder(nn.Module):
    '''
    Decoder model for trajectory prediction
    Based on the paper: https://arxiv.org/pdf/2011.04943.pdf
    :param input_size: input size of the model = 256
    :param hidden_size: hidden size of the model = 512
    :param output_size: output size of the model = 4
    :param frames_future: future frames we want to predict = 45
    :param num_layers: number of layers in the model = 1

    From paper: Input = (batch, 8) where batch is the number of frames in the trajectory
    Output of LSTM will be fed to a fully connected layer with ReLu activation
    We run the LSTM p times to reproduce p hidden states
    Output of the fully connected layer will be of shape (p, 4)
    '''

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 frames_future, 
                 num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p = frames_future
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, enc_hs: torch.Tensor) -> torch.Tensor:
        cell_state = torch.zeros_like(enc_hs)
        # x => [N, I] => [N, 1, I]
        _, (dec_total_output, cell_state) = self.lstm(x, (enc_hs, cell_state))
        dec_h = dec_total_output

        for i in range(self.p - 1):
            _, (dec_h, cell_state) = self.lstm(x, (dec_h, cell_state))
            dec_total_output = torch.cat((dec_total_output, dec_h), dim=0)

        dec_total_output = dec_total_output.transpose(0, 1)
        dec_total_output = self.relu(dec_total_output)
        dec_total_output = self.fc(dec_total_output)
        return dec_total_output

class Predictor(nn.Module):

    def __init__(self, 
                 input_size = 8, 
                 encoder_hidden_size = 512, 
                 encoder_output_size = 256, 
                 num_layers=1, 
                 decoder_hidden_size = 512,
                 frames_future = 60, 

                 ):

        self.enc = TrajPred(input_size, encoder_hidden_size, encoder_output_size, num_layers)
        self.dec = Decoder(encoder_output_size, decoder_hidden_size, input_size, frames_future, num_layers)

    def forward(self, x):
        enc_output, dec_output, enc_h = self.enc(x)
        dec_output = self.dec(enc_output, enc_h)
        return dec_output

class TrajConcat(nn.Module):

    def __init__(self, args = None, velocity = True) -> None:
        super().__init__()
        self.args = args
        self.velocity = velocity

    def forward(self, x, last_bb):
        if self.velocity or self.args.velocity:
            # Indexing with arrays to perserve the shape
            # Do note that because we reversed in flipud, we should use the first bb
            last_bb = last_bb[:, [0], :4]
        assert x.shape[0] == last_bb.shape[0] and x.shape[-1] == last_bb.shape[-1] \
            , f"x and last bounding box should be of same shape, Instead got {x.shape} and {last_bb.shape}"
        return last_bb + x

class Loss(nn.Module):

    def __init__(self, args, alpha = 1.0, beta = 2.0) -> None:
        super().__init__()
        self.args = args
        self.alpha = alpha
        self.beta = beta

    def forward(self, dec_bb, inp_vec, out_bb, pred_bb):
        """
        :param dec_bb: decoder output vector in encoder
        :param inp_vec: input vector
        :param out_bb: decoder output vector in encoder
        :param pred_bb: predicted output vector
        """
        loss = nn.L1Loss()
        if self.args.bbox_type == "cxcywh":
            out_bb_xyxy = box_convert(out_bb, in_fmt = 'cxcywh', out_fmt = 'xyxy')
            pred_bb_xyxy = box_convert(pred_bb, in_fmt = 'cxcywh', out_fmt = 'xyxy')
        
        return self.alpha * loss(dec_bb, inp_vec) / self.args.enc_steps / self.args.input_dim  + \
            self.beta * loss(out_bb, pred_bb) / self.args.dec_steps / 4 + \
            distance_box_iou_loss(out_bb_xyxy, pred_bb_xyxy, "mean")

class EncoderLoss(nn.Module):

    def __init__(self, alpha = 1.0, beta = 2.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, dec_bb, inp_vec, ):
        """
        :param dec_bb: decoder output vector in encoder
        :param inp_vec: input vector
        :param out_bb: decoder output vector in encoder
        :param pred_bb: predicted output vector
        """
        loss = nn.L1Loss()
        return self.alpha * loss(dec_bb, inp_vec) / 15 / 4

class DecoderLoss(nn.Module):

    def __init__(self, beta = 2.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, out_bb, pred_bb, ):
        """
        :param dec_bb: decoder output vector in encoder
        :param inp_vec: input vector
        :param out_bb: decoder output vector in encoder
        :param pred_bb: predicted output vector
        """
        loss = nn.L1Loss()
        return self.beta * loss(out_bb, pred_bb) / 30 / 4

class TrajPredGRU(nn.Module):
    '''
    Encoder model for trajectory prediction, replaced LSTM with GRU
    Based on the paper: https://arxiv.org/pdf/2011.04943.pdf
    :param input_size: input size of the model = 8
    :param hidden_size: hidden size of the model = 512
    :param output_size: output size of the model = 256
    :param num_layers: number of layers in the model = 1

    Input would be of shape (batch_size, seq_len, input_size)

    From Paper: input = (k, 8) where k is the number of frames in the trajectory
    Output of LSTM will be fed to a fully connected layer with ReLU activation
    Output of the fully connected layer will be of shape (batch_size, 256)

    Output of the FC layer is fed to another LSTM layer which acts as the decoder
    Run k time to reproduce k hidden states
    '''

    def __init__(self, 
                 input_size = 8, 
                 hidden_size = 512, 
                 output_size = 256, 
                 num_layers=1, 
                 ):
        super(TrajPredGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enc_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, output_size), 
            nn.ReLU(),
        )

        self.dec_gru = nn.GRU(output_size, hidden_size, num_layers, batch_first=True, )
        self.dec_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        k = x.shape[1]

        # Generate a final hidden state vector
        _ , enc_h = self.enc_gru(x)
        init_cell_state = torch.zeros_like(enc_h)
        # enc_h => [D*num_layers, batch_size, hidden_size]
        # we want => [batch_size, ...]
        
        #enc_h_proped = enc_h.squeeze(dim = 1)
        enc_h_proped = enc_h.transpose(0, 1)
        enc_output = self.encoder(enc_h_proped)

        # dec_output = None
        _, dec_h = self.dec_gru(enc_output, enc_h,)
        final_h = dec_h.transpose(0, 1)
        for i in range(k - 1):
            # if dec_output is None:
                # dec_output = self.dec_lstm(enc_output.unsqueeze(1))[0]
                # _ , (dec_h, _) = self.dec_lstm(enc_output)
                # dec_output = dec_h.transpose(0, 1)
                # dec_output = self.dec_lstm(enc_output)[0]
            # else:
                # dec_output = torch.cat((dec_output, self.dec_lstm(enc_output.unsqueeze(1))[0]), dim=1)
            # We reuse previous states from previous iterations of the decoder
            _ , dec_h = self.dec_gru(enc_output, dec_h)
                # dec_output = torch.cat((dec_output, dec_h.transpose(0, 1)), dim=1)
            final_h = torch.cat((final_h, dec_h.transpose(0, 1)), dim=1)

        final_h = self.dec_fc(final_h)

        return enc_output, final_h, enc_h

class DecoderGRU(nn.Module):
    '''
    Decoder model for trajectory prediction
    Based on the paper: https://arxiv.org/pdf/2011.04943.pdf
    :param input_size: input size of the model = 256
    :param hidden_size: hidden size of the model = 512
    :param output_size: output size of the model = 4
    :param frames_future: future frames we want to predict = 45
    :param num_layers: number of layers in the model = 1

    From paper: Input = (batch, 8) where batch is the number of frames in the trajectory
    Output of LSTM will be fed to a fully connected layer with ReLu activation
    We run the LSTM p times to reproduce p hidden states
    Output of the fully connected layer will be of shape (p, 4)
    '''

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 output_size, 
                 frames_future, 
                 num_layers=1):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p = frames_future
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, enc_hs: torch.Tensor) -> torch.Tensor:
        cell_state = torch.zeros_like(enc_hs)
        # x => [N, I] => [N, 1, I]
        _, dec_total_output = self.gru(x, enc_hs)
        dec_h = dec_total_output

        for i in range(self.p - 1):
            _, dec_h  = self.gru(x, dec_h )
            dec_total_output = torch.cat((dec_total_output, dec_h), dim=0)

        dec_total_output = dec_total_output.transpose(0, 1)
        dec_total_output = self.relu(dec_total_output)
        dec_total_output = self.fc(dec_total_output)
        return dec_total_output