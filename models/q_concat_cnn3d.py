import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HIDDEN_SIZE = 128


class QConcatCNN3D(nn.Module):
    """
    Extract features for video and concatenate question embedding when making the prediction.
    C3D-like network architecture for the video stream.
    """

    def __init__(self, batch_size, q_embedding_size, nb_classes, vocab_size):
        super(QConcatCNN3D, self).__init__()

        """ Video stream """
        self.bn_input = nn.BatchNorm3d(3)

        kernel_size = (4, 4, 4)
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=kernel_size, stride=kernel_size)
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3a = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=kernel_size, stride=kernel_size)
        self.bn3 = nn.BatchNorm3d(128)

        self.fc6 = nn.Linear(7680, 2048)
        self.bn6 = nn.BatchNorm1d(2048)
        self.fc7 = nn.Linear(2048, HIDDEN_SIZE)
        self.bn7 = nn.BatchNorm1d(HIDDEN_SIZE)

        """ Question stream """
        self.batch_size = batch_size
        self.embed = nn.Embedding(vocab_size, q_embedding_size)
        self.q_lstm = nn.LSTM(q_embedding_size, HIDDEN_SIZE)
        self.init_hidden()

        """ Classifier """
        self.fc_tail = nn.Linear(2 * HIDDEN_SIZE, 2 * HIDDEN_SIZE)
        self.d_tail = nn.Dropout(0.5)
        self.out_linear = nn.Linear(2 * HIDDEN_SIZE, nb_classes)

        self.relu = nn.ReLU(inplace=True)

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
        self.q_hidden1 = (Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)),
                         Variable(torch.zeros(1, self.batch_size, HIDDEN_SIZE)))
        if torch.cuda.is_available():
            self.q_hidden1 = (self.q_hidden1[0].cuda(), self.q_hidden1[1].cuda())


    def forward(self, v_input, q_input, v_lens, q_lens, actions=None):
        """ Video feature extraction """
        v_input = self.bn_input(v_input)

        h = self.relu(self.conv1(v_input))
        h = self.pool1(h)
        h = self.bn1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        h = self.bn2(h)

        h = self.relu(self.conv3a(h))
        h = self.pool3(h)
        h = self.bn3(h)

        h = h.view(h.shape[0], -1)

        h = self.relu(self.fc6(h))
        h = self.bn6(h)
        h = self.relu(self.fc7(h))
        h = self.bn7(h)

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
        out = torch.cat([h, q_final_timestep], 1)
        """ Classify """
        out = self.fc_tail(out)
        out = self.relu(out)
        out = self.d_tail(out)
        return self.out_linear(out)
