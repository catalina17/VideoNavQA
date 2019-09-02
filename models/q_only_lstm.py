import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class QOnlyLSTM(nn.Module):
    """
    Question-only LSTM model.
    """
    def __init__(self, batch_size, embedding_size, hidden_size, nb_classes, vocab_size):
        super(QOnlyLSTM, self).__init__()

        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.init_hidden()

        self.out_linear = nn.Linear(hidden_size, nb_classes)

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
        self.hidden_1 = (Variable(torch.randn(1, self.batch_size, self.hidden_size)),
                         Variable(torch.randn(1, self.batch_size, self.hidden_size)))
        if torch.cuda.is_available:
            self.hidden_1 = (self.hidden_1[0].cuda(), self.hidden_1[1].cuda())


    def forward(self, q_input, q_lens):
        q_embed_input = self.embed(q_input)
        q_embed_input = q_embed_input.permute(1, 0, 2)

        packed_q_embed_input = pack_padded_sequence(q_embed_input, q_lens.cpu().numpy())
        packed_lstm_out, self.hidden_1 = self.lstm(packed_q_embed_input, self.hidden_1)
        lstm_out = pad_packed_sequence(packed_lstm_out)
        lstm_out = lstm_out[0].permute(1, 0, 2)

        idx = q_lens.view(self.batch_size, 1, 1).expand(self.batch_size, 1, self.hidden_size) - 1
        final_timestep = lstm_out.gather(1, idx).view(self.batch_size, self.hidden_size)

        return self.out_linear(final_timestep)
