import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class QOnlyBOW(nn.Module):
    """
    Question-only averaged bag-of-words model.
    """
    def __init__(self, batch_size, embedding_size, nb_classes, vocab_size):
        super(QOnlyBOW, self).__init__()

        self.nb_classes = nb_classes
        self.batch_size = batch_size

        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.out_linear = nn.Linear(embedding_size, nb_classes)

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


    def forward(self, q_input, q_lens):
        q_embed_input = self.embed(q_input)

        q_embed_average = torch.sum(q_embed_input, dim=1)
        for i in range(self.batch_size):
            torch.div(q_embed_average[i], q_lens[i].type(torch.cuda.FloatTensor))

        return self.out_linear(q_embed_average)
