import torch
import torch.nn as nn


class ObjDetectCNN(nn.Module):
    """
    2D CNN for detecting objects in video frames from the VideoNavQA dataset. Takes as input visual
    features from a pre-trained Faster R-CNN (VGG16).
    """

    def __init__(self,
                 nb_classes,
                 num_filters=128,
                 tail_hidden_dim=256,
                 tail_dropout_p=0.5,
                 logits=False,
                 pretrained_features=False):
        super(ObjDetectCNN, self).__init__()
        self.logits = logits
        self.pretrained_features = pretrained_features

        self.bn_input = nn.BatchNorm2d(128)

        self.conv11 = nn.Conv2d(128, num_filters, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv31 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_tail1 = nn.Linear(num_filters * 6 * 5, tail_hidden_dim)
        self.bn_tail1 = nn.BatchNorm1d(tail_hidden_dim)
        self.fc_tail2 = nn.Linear(tail_hidden_dim, nb_classes)

        self.dropout = nn.Dropout(p=tail_dropout_p)
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


    def forward(self, inputs):
        inputs = self.bn_input(inputs)

        h = self.conv12(self.conv11(inputs))
        h = self.bn1(h)
        h = self.relu(h)
        h = self.pool1(h)

        h = self.conv22(self.conv21(h))
        h = self.bn2(h)
        h = self.relu(h)
        h = self.pool2(h)

        h = self.conv32(self.conv31(h))
        h = self.bn3(h)
        h = self.relu(h)
        if self.pretrained_features:
            return h

        h = self.pool3(h)

        h = h.view(h.shape[0], -1)

        h = self.fc_tail1(h)
        h = self.bn_tail1(h)
        h = self.relu(h)

        res = self.fc_tail2(h)
        if self.logits:
            return res
        return torch.sigmoid(res)
