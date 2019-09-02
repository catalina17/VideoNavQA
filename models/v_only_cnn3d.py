import torch
import torch.nn as nn


class VideoOnlyCNN3D(nn.Module):
    """
    C3D-like network architecture
    Tran, Du, et al. "Learning spatiotemporal features with 3D convolutional networks." ICCV 2015.
    """

    def __init__(self, nb_classes):
        super(VideoOnlyCNN3D, self).__init__()
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
        self.fc7 = nn.Linear(2048, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.fc8 = nn.Linear(128, nb_classes)

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

        h = self.relu(self.conv1(inputs))
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

        return self.fc8(h)
