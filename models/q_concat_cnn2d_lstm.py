import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models.vgg import make_layers

HIDDEN_SIZE = 128


class QConcatCNN2DLSTM(nn.Module):
    """
    Extract features for each frame and concatenate question embedding when making the prediction.
    CNN 2D per-frame feature extractor is a VGG 11-layer architecture.
    """
    def __init__(self, batch_size, q_embedding_size, nb_classes, vocab_size):
        super(QConcatCNN2DLSTM, self).__init__()

        self.batch_size = batch_size
        self.use_actions = use_actions

        """ Video stream """
        # Per-frame extractor
        cfg = [16, 'M', 32, 'M', 64, 'M', 128, 'M', 128, 'M']
        self.per_frame_feature_extractor = make_layers(cfg, batch_norm=True)

        # Tail classifier taking features from all frames
        self.v_lstm = nn.LSTM(cfg[-2] * 5 * 6, HIDDEN_SIZE)

        """ Question embedding """
        self.embed = nn.Embedding(vocab_size, q_embedding_size)
        self.q_lstm = nn.LSTM(q_embedding_size, HIDDEN_SIZE)
        self.init_hidden()

        """ Classifier """
        self.fc_tail = nn.Linear(2 * HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        self.relu = nn.ReLU(inplace=True)
        self.d_tail = nn.Dropout(0.5)
        self.out_linear = nn.Linear(2 * HIDDEN_SIZE, nb_classes)

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
        self.v_hidden1 = (Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)),
                         Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)))
        self.q_hidden1 = (Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)),
                         Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)))
        if torch.cuda.is_available():
            self.v_hidden1 = (self.v_hidden1[0].cuda(), self.v_hidden1[1].cuda())
            self.q_hidden1 = (self.q_hidden1[0].cuda(), self.q_hidden1[1].cuda())


    """
    Describes the forward propagation method for this model. Assumes the input videos are sorted
    by length in descending order.
    """
    def forward(self, v_input, q_input, v_lens, q_lens, actions=None):
        """ Video feature extraction """
        num_frames = v_input.shape[-1]

        dim = 128 * 5 * 6 + int(self.use_actions)
        v_features = torch.zeros(num_frames, v_input.shape[0], dim)
        if torch.cuda.is_available():
            v_features = v_features.cuda()

        ct_batch_size = v_input.shape[0]
        for i in range(num_frames):
            # Find out effective batch size at current timestep, i.e. how many videos are this long
            while v_lens[ct_batch_size - 1] < (i + 1) and ct_batch_size >= 0:
                ct_batch_size -= 1

            # No more videos of this length
            if ct_batch_size == -1:
                break

            frame_features = self.per_frame_feature_extractor(v_input[:ct_batch_size, :, :, :, i])
            if self.use_actions:
                action_emb = self.action_embed(actions[:ct_batch_size, i])
                v_features[i, :ct_batch_size, :] = torch.cat(
                    (frame_features.view(frame_features.size(0), -1), action_emb), 1)
            else:
                v_features[i, :ct_batch_size, :] = frame_features.view(frame_features.size(0), -1)

        packed_v_features = pack_padded_sequence(v_features, v_lens.cpu().numpy())
        packed_v_lstm_out, self.v_hidden1 = self.v_lstm(packed_v_features, self.v_hidden1)
        v_lstm_out = pad_packed_sequence(packed_v_lstm_out)
        v_lstm_out = v_lstm_out[0].permute(1, 0, 2)

        idx = v_lens.view(self.batch_size, 1, 1).expand(self.batch_size, 1, HIDDEN_SIZE) - 1
        v_final_timestep = v_lstm_out.gather(1, idx).view(self.batch_size, HIDDEN_SIZE)

        """ Question embedding """
        q_embed_input = self.embed(q_input)

        init_lens = q_lens
        q_lens, perm_idx = q_lens.sort(0, descending=True)
        q_embed_input = q_embed_input[perm_idx]
        q_embed_input = q_embed_input.permute(1, 0, 2)

        packed_q_embed_input = pack_padded_sequence(q_embed_input, q_lens.cpu().numpy())
        packed_q_lstm_out, self.q_hidden1 = self.q_lstm(packed_q_embed_input, self.q_hidden1)
        q_lstm_out = pad_packed_sequence(packed_q_lstm_out)

        _, invperm_idx = perm_idx.sort(0)
        q_lstm_out = q_lstm_out[0]
        q_lstm_out = q_lstm_out.permute(1, 0, 2)
        q_lstm_out = q_lstm_out[invperm_idx]

        idx = init_lens.view(self.batch_size, 1, 1).expand(self.batch_size, 1, HIDDEN_SIZE) - 1
        q_final_timestep = q_lstm_out.gather(1, idx).view(self.batch_size, HIDDEN_SIZE)

        # Concatenate the two modalities
        out = torch.cat([v_final_timestep, q_final_timestep], 1)
        """ Classify """
        out = self.fc_tail(out)
        out = self.relu(out)
        out = self.d_tail(out)
        return self.out_linear(out)
