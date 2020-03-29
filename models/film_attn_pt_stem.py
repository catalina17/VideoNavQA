import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FiLMAttnPretrainedStem(nn.Module):
    """
    CNN 2D per-frame feature extractor is a VGG 11-layer architecture. Per-frame question
    conditioning via FiLM layers. Attention at the tail over the frame features.
    """
    def __init__(self,
                 batch_size,
                 q_embedding_size,
                 nb_classes,
                 num_input_channels=512,
                 num_res_block_channels=512,
                 num_res_blocks=1,
                 hidden_size=128,
                 at_hidden_size=128,
                 max_num_frames=35,
                 q_encoder='lstm',
                 vocab_size=134):
        super(FiLMAttnPretrainedStem, self).__init__()

        assert q_encoder.lower() in ['lstm', 'bow'], "Invalid question encoder! (\'lstm\', \'bow\')"
        self.q_encoder = q_encoder

        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.q_embedding_size = q_embedding_size
        self.at_hidden_size = at_hidden_size
        self.hidden_size = hidden_size
        # Embedding layer for question input
        self.embed = nn.Embedding(vocab_size, q_embedding_size)

        self.relu = nn.ReLU(inplace=True)
        self.conv_init = nn.Conv2d(num_input_channels, num_res_block_channels,
                                   kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(num_res_block_channels)

        self.conv1x1_layers = [] # initialized in get_film_pipeline()

        """ FiLM pipeline """
        total_out_size = 0
        film_cfg = [num_res_block_channels] * num_res_blocks
        for item in film_cfg:
            total_out_size += 2 * item
        self.film_layer = self.film_encoder_decoder(hidden_size, total_out_size)
        self.film_pipeline = self.get_film_pipeline(film_cfg, num_res_block_channels)

        """ Tail attention mechanism """
        # Data embedding layer
        dim = 130 * num_res_block_channels
        self.fc_embed_attn = nn.Linear(dim, at_hidden_size)
        self.fc_attn_1 = nn.Linear(at_hidden_size, 1)
        # MLP for LSTM hidden state - 2nd half of attention mechanism
        self.fc_hidden_attn = nn.Linear(at_hidden_size, 1)
        # LSTM - takes context and hidden state at each step
        self.lstm_attn = nn.LSTMCell(at_hidden_size, at_hidden_size)

        # Classification layer
        self.out_linear = nn.Linear(max_num_frames * at_hidden_size, nb_classes)

        for module in self.modules():
            self.weights_init(module)
        self.init_hidden()


    """
    Returns a list of layers that corresponds to a FiLM generator for the input question.
    """
    def film_encoder_decoder(self, nb_hidden, nb_out):
        encoder_layer = nn.LSTM(self.q_embedding_size, nb_hidden) if self.q_encoder == 'lstm' else\
                        nn.Linear(self.q_embedding_size, nb_hidden)
        layers = [
            encoder_layer,
            nn.Linear(nb_hidden, nb_out),
            nn.ReLU(inplace=True),
        ]

        if torch.cuda.is_available():
            layers = nn.ModuleList(layers)
            layers = layers.cuda()
        return layers


    """
    Returns a list of layers for the FiLM pipeline.
    """
    def get_film_pipeline(self, cfg, num_in_channels):
        layers = []
        in_channels = num_in_channels

        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.append(conv2d)

            conv1x1 = nn.Conv2d(in_channels, v, kernel_size=1)
            if torch.cuda.is_available():
                conv1x1 = conv1x1.cuda()
            self.conv1x1_layers.append(conv1x1)

            in_channels = v

        return nn.ModuleList(layers)


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
        if self.q_encoder == 'lstm':
            self.film_hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                                Variable(torch.zeros(1, self.batch_size, self.hidden_size)))
            if torch.cuda.is_available():
                self.film_hidden = (self.film_hidden[0].cuda(), self.film_hidden[1].cuda())


    """
    Calculates the FiLM conditional values from the question embedding.
    """
    def compute_film_values(self, q_input, q_lens, ct_batch_size):
        # Question embedding
        x = self.embed(q_input)

        if self.q_encoder == 'lstm':
            init_lens = q_lens
            q_lens, perm_idx = q_lens.sort(0, descending=True)
            x = x[perm_idx]
            x = x.permute(1, 0, 2)

        for i in range(len(self.film_layer)):
            op = self.film_layer[i]
            if i == 0:
                # Question encoder can be LSTM or BoW model
                if isinstance(op, nn.LSTM):
                    packed_x = pack_padded_sequence(x, q_lens.cpu().numpy())
                    packed_lstm_op_out, self.film_hidden = op(packed_x, self.film_hidden)
                    lstm_op_out = pad_packed_sequence(packed_lstm_op_out)

                    _, invperm_idx = perm_idx.sort(0)
                    lstm_op_out = lstm_op_out[0]
                    lstm_op_out = lstm_op_out.permute(1, 0, 2)
                    lstm_op_out = lstm_op_out[invperm_idx]

                    idx = init_lens.view(self.batch_size, 1, 1).expand(
                        self.batch_size, 1, op.hidden_size) - 1
                    x = lstm_op_out.gather(1, idx).view(
                        self.batch_size, op.hidden_size)[:ct_batch_size, :]
                else:
                    x = op(x)
                    x = torch.sum(x, dim=1)
                    for j in range(self.batch_size):
                        torch.div(x[j], q_lens[j].type(torch.cuda.FloatTensor))
                    x = x[:ct_batch_size, :]
            else:
                x = op(x)

        return x


    """
    Describes the forward propagation method for this model. Assumes the input videos are sorted
    by length in descending order.
    """
    def forward(self, v_input, q_input, v_lens, q_lens):
        """ Video feature extraction """
        num_frames = v_input.shape[-1]
        actual_num_frames = v_lens[0]
        all_features = []

        masks = torch.zeros(v_input.shape[0], num_frames, 1) # for attention
        if torch.cuda.is_available():
            masks = masks.cuda()

        ct_batch_size = v_input.shape[0]
        ct_batch_sizes = np.empty(num_frames, dtype=np.int32)

        for i in range(num_frames):
            # Find out effective batch size at current timestep, i.e. how many videos are this long
            while ct_batch_size >= 0 and v_lens[ct_batch_size - 1] < (i + 1):
                ct_batch_size -= 1

            # No more videos of this length
            if ct_batch_size == -1:
                break

            v_features = v_input[:ct_batch_size, :, :, :, i]
            v_features = self.bn_init(self.relu(self.conv_init(v_features)))
            # Get FiLM conditional values
            film_values = self.compute_film_values(q_input, q_lens, ct_batch_size)

            start_idx = 0
            block_idx = 0
            for layer in self.film_pipeline:
                # Compute 1x1 convolution that is skip-connected to the end of the block
                res_x = self.relu(self.conv1x1_layers[block_idx](v_features))
                block_idx += 1
                v_features = res_x

                # Compute the 3x3 convolution
                v_features = layer(v_features)

                # Compute the new feature maps - v_features.shape[1] gives their number
                # First compute FiLM alphas and betas
                s = v_features.shape
                delim_idx = (start_idx + v_features.shape[1])

                alphas = film_values[:, start_idx : delim_idx]
                betas = film_values[:, delim_idx : delim_idx + v_features.shape[1]]
                start_idx += 2 * v_features.shape[1]

                alphas = alphas.expand(s[2], s[3], s[0], s[1]).permute(2, 3, 0, 1)
                betas = betas.expand(s[2], s[3], s[0], s[1]).permute(2, 3, 0, 1)

                # Compute FiLM transformation + ReLU activation
                v_features = self.relu(alphas * v_features + betas)
                # Compute final result of ResBlock
                v_features = v_features + res_x

            # Prepare visual features for attention
            v_features = self.fc_embed_attn(v_features.view(v_features.size(0), -1))
            padding = (0, 0, 0, self.batch_size - ct_batch_size)
            all_features.append(F.pad(v_features.view(1, v_features.size(0), -1),
                                      padding, 'constant'))

            ct_batch_sizes[i] = ct_batch_size
            # Fill corresponding region of mask with -INF
            masks[ct_batch_size:, i, 0].fill_(-(1 << 31))

        all_features = torch.cat(all_features, dim=0)
        all_features = all_features.permute(1, 0, 2)
        padding = (0, 0, 0, num_frames - all_features.shape[1])
        all_features = F.pad(all_features, padding, 'constant')

        h = Variable(torch.zeros(self.batch_size, 1, self.at_hidden_size))
        hs = Variable(torch.zeros(self.batch_size, num_frames, self.at_hidden_size))
        cell = Variable(torch.zeros(self.batch_size, self.at_hidden_size))
        ctxt = Variable(torch.zeros(self.batch_size, self.at_hidden_size))
        if torch.cuda.is_available():
            h, hs, cell, ctxt = h.cuda(), hs.cuda(), cell.cuda(), ctxt.cuda()

        """ Attention over feature vectors from all frames """
        # Flatten feature vectors (to fix variable video count across frames) for fc_attn layer
        # i.e. B * T * HIDDEN_SIZE -> (\sum_{t = 1}^{T} nvids_{T}) * HIDDEN_SIZE
        features_BxT = []
        for i in range(actual_num_frames):
            features_BxT.append(all_features[:ct_batch_sizes[i], i, :])
        features_BxT = torch.cat(features_BxT, dim=0)
        features_BxT = self.fc_attn_1(features_BxT) # 1st half of attention

        # Reshape new feature vector to bring back to matrix form; will use masks for attention
        # i.e. (\sum_{t = 1}^{T} nvids_{T}) * 1 -> B * T * 1
        features = torch.zeros(self.batch_size, num_frames, 1)
        if torch.cuda.is_available():
            features = features.cuda()
        for i in range(actual_num_frames):
            features[:ct_batch_sizes[i], i] =\
                features_BxT[ct_batch_sizes[:i].sum() : ct_batch_sizes[:(i + 1)].sum()]

        for i in range(num_frames):
            # Encode hidden state, replicate for all frames in the video
            v_i = self.fc_hidden_attn(h).repeat(1, num_frames, 1)

            # Calculate attention coefficients
            coefs = nn.Softmax(dim=1)(v_i + features + masks)
            # Obtain weighted result - context for the LSTM cell
            ctxt = torch.bmm(coefs.permute(0, 2, 1), all_features).view(self.batch_size, -1)

            # Get new context and hidden state
            h, cell = self.lstm_attn(ctxt, (h.view(self.batch_size, -1), cell))
            hs[:, i, :] = h
            h = h.view(self.batch_size, +1, self.at_hidden_size)

        # Attended output
        hs = hs.view(self.batch_size, hs.shape[1] * hs.shape[2])

        """ Classifier """
        return self.out_linear(hs)
