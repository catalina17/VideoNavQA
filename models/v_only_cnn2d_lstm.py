import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models.vgg import make_layers

HIDDEN_SIZE = 128


class VideoOnlyCNN2DLSTM(nn.Module):
    """
    CNN 2D per-frame feature extractor is a VGG 11-layer architecture.
    """
    def __init__(self, batch_size, nb_classes):
        super(VideoOnlyCNN2DLSTM, self).__init__()

        self.batch_size = batch_size
        self.input_bn = nn.BatchNorm3d(3)

        # Per-frame extractor
        cfg = [16, 'M', 32, 'M', 64, 'M', 128, 'M', 128, 'M']
        self.per_frame_feature_extractor = make_layers(cfg, batch_norm=True)

        # Tail classifier taking features from all frames
        self.lstm = nn.LSTM(cfg[-2] * 5 * 6, HIDDEN_SIZE)
        self.init_hidden()

        # Classification layer
        self.out_linear = nn.Linear(HIDDEN_SIZE, nb_classes)

        for module in self.modules():
            self.weights_init(module)


    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)

        if isinstance(m, nn.LSTM):
            nn.init.xavier_uniform_(m.weight_ih_l0)
            nn.init.orthogonal_(m.weight_hh_l0)

            for names in m._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)

            m.bias_ih_l0.data.fill_(0.0)


    """
    Initializes the hidden states of LSTM layers.
    """
    def init_hidden(self):
        self.hidden_1 = (Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)),
                         Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)))
        if torch.cuda.is_available():
            self.hidden_1 = (self.hidden_1[0].cuda(), self.hidden_1[1].cuda())


    """
    Describes the forward propagation method for this model. Assumes the input videos are sorted
    by length in descending order.
    """
    def forward(self, video_frames, video_lengths):
        video_frames = self.input_bn(video_frames)

        num_frames = video_frames.shape[-1]
        v_features = torch.zeros(num_frames, video_frames.shape[0], 128 * 5 * 6)
        if torch.cuda.is_available():
            v_features = v_features.cuda()

        ct_batch_size = video_frames.shape[0]
        # Video feature extraction
        for i in range(num_frames):
            # Find out effective batch size at current timestep, i.e. how many videos are this long
            while video_lengths[ct_batch_size - 1] < (i + 1) and ct_batch_size >= 0:
                ct_batch_size -= 1

            # No more videos of this length
            if ct_batch_size == -1:
                break

            frame_features = self.per_frame_feature_extractor(
                video_frames[:ct_batch_size, :, :, :, i])
            v_features[i, :ct_batch_size, :] = frame_features.view(frame_features.size(0), -1)

        packed_v_features = pack_padded_sequence(v_features, video_lengths.cpu().numpy())

        packed_lstm_out, self.hidden_1 = self.lstm(packed_v_features, self.hidden_1)

        lstm_out = pad_packed_sequence(packed_lstm_out)
        lstm_out = lstm_out[0].permute(1, 0, 2)

        idx = video_lengths.view(self.batch_size, 1, 1).expand(self.batch_size, 1, HIDDEN_SIZE) - 1
        final_timestep = lstm_out.gather(1, idx).view(self.batch_size, HIDDEN_SIZE)

        return self.out_linear(final_timestep)
