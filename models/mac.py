import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin

class ControlUnit(nn.Module):
    def __init__(self, dim, max_step):
        super(ControlUnit, self).__init__()

        self.position_aware = nn.ModuleList()
        for i in range(max_step):
            self.position_aware.append(linear(dim * 2, dim))

        self.control_question = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

        self.dim = dim

    def forward(self, step, context, question, control):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)
        control_question = control_question.unsqueeze(1)

        context_prod = control_question * context
        attn_weight = self.attn(context_prod)

        attn = F.softmax(attn_weight, 1)

        next_control = (attn * context).sum(1)

        return next_control


class ReadUnit(nn.Module):
    def __init__(self, dim):
        super(ReadUnit, self).__init__()

        self.mem = linear(dim, dim)
        self.concat = linear(dim * 2, dim)
        self.attn = linear(dim, 1)

    def forward(self, memory, know, control):
        mem = self.mem(memory[-1]).unsqueeze(2)
        concat = self.concat(torch.cat([mem * know, know], 1) \
                                .permute(0, 2, 1))
        attn = concat * control[-1].unsqueeze(1)
        attn = self.attn(attn).squeeze(2)
        attn = F.softmax(attn, 1).unsqueeze(1)

        read = (attn * know).sum(2)

        return read


class WriteUnit(nn.Module):
    def __init__(self, dim, self_attention=False, memory_gate=False):
        super(WriteUnit, self).__init__()

        self.concat = linear(dim * 2, dim)

        if self_attention:
            self.attn = linear(dim, 1)
            self.mem = linear(dim, dim)

        if memory_gate:
            self.control = linear(dim, 1)

        self.self_attention = self_attention
        self.memory_gate = memory_gate

    def forward(self, memories, retrieved, controls):
        prev_mem = memories[-1]
        concat = self.concat(torch.cat([retrieved, prev_mem], 1))
        next_mem = concat

        if self.self_attention:
            controls_cat = torch.stack(controls[:-1], 2)
            attn = controls[-1].unsqueeze(2) * controls_cat
            attn = self.attn(attn.permute(0, 2, 1))
            attn = F.softmax(attn, 1).permute(0, 2, 1)

            memories_cat = torch.stack(memories, 2)
            attn_mem = (attn * memories_cat).sum(2)
            next_mem = self.mem(attn_mem) + concat

        if self.memory_gate:
            control = self.control(controls[-1])
            gate = F.sigmoid(control)
            next_mem = gate * prev_mem + (1 - gate) * next_mem

        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12,
                self_attention=False, memory_gate=False,
                dropout=0.15):
        super(MACUnit, self).__init__()

        self.control = ControlUnit(dim, max_step)
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim, self_attention, memory_gate)

        self.mem_0 = nn.Parameter(torch.zeros(1, dim))
        self.control_0 = nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

    def get_mask(self, x, dropout):
        mask = torch.empty_like(x).bernoulli_(1 - dropout)
        mask = mask / (1 - dropout)

        return mask

    def forward(self, context, question, knowledge):
        b_size = question.size(0)

        control = self.control_0.expand(b_size, self.dim)
        memory = self.mem_0.expand(b_size, self.dim)

        if self.training:
            control_mask = self.get_mask(control, self.dropout)
            memory_mask = self.get_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        controls = [control]
        memories = [memory]

        for i in range(self.max_step):
            control = self.control(i, context, question, control)
            if self.training:
                control = control * control_mask
            controls.append(control)

            read = self.read(memories, knowledge, controls)
            memory = self.write(memories, read, controls)
            if self.training:
                memory = memory * memory_mask
            memories.append(memory)

        return memory


class MACNetwork(nn.Module):
    def __init__(self, n_vocab, dim, embed_hidden=300,
                max_step=12, self_attention=False, memory_gate=False,
                classes=28, dropout=0.15, max_num_frames=35):
        super(MACNetwork, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(512, dim, 3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.ELU(),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.ELU())

        self.embed = nn.Embedding(n_vocab, embed_hidden, padding_idx=0)
        self.lstm = nn.LSTM(embed_hidden, dim,
                        batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(dim * 2, dim)

        self.mac = MACUnit(dim, max_step,
                        self_attention, memory_gate, dropout)


        self.lstm_tail = nn.LSTM(dim * 3, dim * 3)
        self.classifier = nn.Sequential(linear(dim * 3, dim * 2),
                                        nn.ELU(),
                                        linear(dim * 2, classes))

        self.max_step = max_step
        self.dim = dim
        self.max_num_frames = max_num_frames

        self.reset()

    def reset(self):
        self.embed.weight.data.uniform_(0, 1)

        kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        kaiming_uniform_(self.classifier[0].weight)

    def forward(self, images, question, v_lens, question_len, actions=None, dropout=0.15):
        b_size = images.size(0)

        # Process question
        embed = self.embed(question[:b_size])

        question_len, perm_idx = question_len.sort(0, descending=True)
        embed = embed[perm_idx]

        embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
                                                  batch_first=True)
        lstm_out, (h, _) = self.lstm(embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
                                                       batch_first=True)

        _, invperm_idx = perm_idx.sort(0)
        lstm_out = lstm_out[invperm_idx]

        lstm_out = self.lstm_proj(lstm_out)
        h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

        # Per frame processing for input to MAC chain (with the same question embedding)
        outs = []
        for i in range(v_lens[0]):
            # Find out effective batch size at current timestep, i.e. how many videos are this long.
            while b_size >= 0 and v_lens[b_size - 1] < (i + 1):
                b_size -= 1

            # No more videos of this length
            if b_size == -1:
                break

            image = images[:b_size, :, :, :, i]
            img = self.conv(image)
            img = img.view(b_size, self.dim, -1)

            memory = self.mac(lstm_out[:b_size], h[:b_size], img)
            out = torch.cat([memory, h[:b_size]], 1)

            padding = (0, 0, 0, v_lens.size(0) - b_size)
            outs.append(F.pad(out, padding, 'constant').view(v_lens.size(0), 1, out.size(-1)))

        outs = torch.cat(outs, dim=1)
        padding = (0, 0, 0, self.max_num_frames - v_lens[0])
        outs = F.pad(outs, padding, 'constant')

        outs = nn.utils.rnn.pack_padded_sequence(outs, v_lens,
                                                 batch_first=True)
        lstm_out_tail, (_, _) = self.lstm_tail(outs)
        lstm_out_tail, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_tail,
                                                            batch_first=True)
        idx = v_lens.view(v_lens.size(0), 1, 1).expand(
            v_lens.size(0), 1, lstm_out_tail.size(-1)) - 1
        out = lstm_out_tail.gather(1, idx).view(v_lens.size(0), lstm_out_tail.size(-1))
        out = self.classifier(out)

        return out
