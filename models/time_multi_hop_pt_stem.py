import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TimeMultiHopFiLMPretrainedStem(nn.Module):
    """
    FiLM model leveraging multi-hop question embedding across time. Global max-pooling for temporal
    summarization. Takes as input visual features from a pre-trained Faster R-CNN (VGG16).
    """
    def __init__(self,
                 batch_size,
                 q_embedding_size,
                 nb_classes,
                 num_input_channels=512,
                 num_res_block_channels=512,
                 num_res_blocks=1,
                 num_tail_channels=32,
                 hidden_size=128,
                 vocab_size):
        super(TimeMultiHopFiLMPretrainedStem, self).__init__()

        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.q_embedding_size = q_embedding_size
        self.hidden_size = hidden_size
        # Embedding layer for question input
        self.embed = nn.Embedding(vocab_size, q_embedding_size, padding_idx=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv_init = nn.Conv2d(num_input_channels, num_res_block_channels,
                                   kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(num_res_block_channels)

        self.conv1x1_layers = [] # initialized in get_FiLM_pipeline()

        """ FiLM pipeline """
        total_out_size = 0
        film_cfg = [num_res_block_channels] * num_res_blocks
        for item in film_cfg:
            total_out_size += 2 * item

        # FiLM question encoding
        self.q_encoder = nn.LSTM(self.q_embedding_size, hidden_size)
        self.encoder_norm = nn.LayerNorm(hidden_size)
        # FiLM question decodin
        self.h = None # context vector that gets updated across timesteps
        self.fc_hidden_attn = nn.Linear(hidden_size, 1)
        self.fc_attn_out = nn.Linear(hidden_size, total_out_size)
        self.decoder_norm = nn.LayerNorm(total_out_size)

        self.film_pipeline = self.get_FiLM_pipeline(film_cfg, num_res_block_channels)

        """ Tail classifier taking features from all frames """
        self.c1x1_tail = nn.Conv2d(num_res_block_channels, num_tail_channels, kernel_size=1)
        self.init_hidden()

        # Classification layer
        self.out_linear = nn.Linear(130 * num_tail_channels, nb_classes)

        for module in self.modules():
            self.weights_init(module)
        for module in self.conv1x1_layers:
            self.weights_init(module)


    """
    Returns a list of layers for the FiLM pipeline.
    """
    def get_FiLM_pipeline(self, cfg, num_in_channels):
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
        self.film_hidden = (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                            Variable(torch.zeros(1, self.batch_size, self.hidden_size)))
        if torch.cuda.is_available():
            self.film_hidden = (self.film_hidden[0].cuda(), self.film_hidden[1].cuda())
        self.h = None


    """
    Computes the FiLM encoding from the question embedding - the RNN hidden states across time and
    the final state will be saved for computing the FiLM values at each timestep. They will depend
    on the context from the previous timestep, which then gets updated.
    """
    def compute_film_encoding(self, q_input, q_lens, ct_batch_size):
        # Question embedding
        x = self.embed(q_input)

        # Question encoding for FiLM generator
        init_lens = q_lens
        q_lens, perm_idx = q_lens.sort(0, descending=True)
        x = x[perm_idx]
        x = x.permute(1, 0, 2)

        packed_x = pack_padded_sequence(x, q_lens.cpu().numpy())
        packed_lstm_op_out, self.film_hidden = self.q_encoder(packed_x, self.film_hidden)
        lstm_op_out = pad_packed_sequence(packed_lstm_op_out)

        _, invperm_idx = perm_idx.sort(0)
        lstm_op_out = lstm_op_out[0]
        lstm_op_out = lstm_op_out.permute(1, 0, 2)
        lstm_op_out = lstm_op_out[invperm_idx]

        idx = init_lens.view(self.batch_size, 1, 1).expand(
            self.batch_size, 1, self.q_encoder.hidden_size) - 1
        x = lstm_op_out.gather(1, idx).view(self.batch_size, self.q_encoder.hidden_size)

        x = x[:ct_batch_size, :]
        x = self.encoder_norm(x)

        # Question decoding for FiLM generator
        encoded, self.rnn_states = x, lstm_op_out

        self.num_words = lstm_op_out.size(1)
        self.film_out_size = self.decoder_norm.normalized_shape[0]

        # Initialize context vector with final state of RNN encoder or take only current batch size
        self.h = encoded.view(ct_batch_size, 1, self.q_encoder.hidden_size).clone().\
                         repeat(1, self.num_words, 1)


    """
    Computes the FiLM values from the question embedding, for the current timestep. See method
    description for compute_film_encoding().
    """
    def decode_to_film_values(self, ct_batch_size):
        self.h = self.h[:ct_batch_size, :, :]
        self.rnn_states = self.rnn_states[:ct_batch_size, :, :]

        # Calculate raw coefs: B x T x H -> B x T x 1 -> B x 1 x T
        element_wise_prod = self.h * self.rnn_states
        coefs = self.fc_hidden_attn(element_wise_prod)
        coefs = nn.Softmax(dim=1)(coefs).view(ct_batch_size, 1, self.num_words)

        # Compute attended vector and get context: (B x 1 x T) * (B x T x H) -> B x 1 x H -> B x H
        self.h = torch.bmm(coefs, element_wise_prod)
        self.h = self.h.view(ct_batch_size, self.q_encoder.hidden_size)

        # Bring context to required dimensionality: B x H -> B x N_out
        h_cond_features = self.fc_attn_out(self.h)
        self.h = self.h.view(ct_batch_size, 1, self.q_encoder.hidden_size).\
            repeat(1, self.num_words, 1)

        # Apply layer normalization and save context for the current module
        return self.decoder_norm(h_cond_features)


    """
    Describes the forward propagation method for this model. Assumes the input videos are sorted
    by length in descending order.
    """
    def forward(self, v_input, q_input, v_lens, q_lens):
        """ Video feature extraction """
        num_frames = v_input.shape[-1]
        all_features_list = []

        ct_batch_size = v_input.shape[0]
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
            self.compute_film_encoding(q_input, q_lens, ct_batch_size)

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

                film_values = self.decode_to_film_values(ct_batch_size)
                alphas = film_values[:, start_idx : delim_idx]
                betas = film_values[:, delim_idx : delim_idx + v_features.shape[1]]
                start_idx += 2 * v_features.shape[1]

                alphas = alphas.expand(s[2], s[3], s[0], s[1]).permute(2, 3, 0, 1)
                betas = betas.expand(s[2], s[3], s[0], s[1]).permute(2, 3, 0, 1)

                # Compute FiLM transformation + ReLU activation
                v_features = self.relu(alphas * v_features + betas)
                # Compute final result of ResBlock
                v_features = v_features + res_x

            v_features = self.relu(self.c1x1_tail(v_features))

            padding = (0, 0, 0, self.batch_size - ct_batch_size)
            all_features_list.append(F.pad(v_features.view(1, v_features.size(0), -1),
                                           padding, 'constant'))

        """ Global temporal pooling across all frames """
        global_max_pooled = torch.max(torch.cat(all_features_list), dim=0)[0]

        """ Classifier """
        return self.out_linear(global_max_pooled)
