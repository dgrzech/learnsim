import math
import torch
import torch.nn as nn

from base import BaseModel


"""
encoding function
"""


class CNN(BaseModel):
    def __init__(self, s):
        super(CNN, self).__init__()

        self.s = s
        self.padding = math.floor(self.s / 2)

        self.conv1 = nn.Conv3d(1, 1, kernel_size=self.s, stride=1, padding=self.padding, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=self.s, stride=1, padding=self.padding, bias=False)

        # initialise kernels to identity
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv2.weight)

        w1 = self.conv1.weight.view(self.s ** 3)
        w1[math.floor((self.s ** 3) / 2)] = 1.0
        w1 = w1.view(1, 1, self.s, self.s, self.s)

        w2 = self.conv2.weight.view(self.s ** 3)
        w2[math.floor((self.s ** 3) / 2)] = 1.0
        w2 = w2.view(1, 1, self.s, self.s, self.s)

        with torch.no_grad():
            self.conv1.weight = nn.Parameter(w1, requires_grad=True)
            self.conv2.weight = nn.Parameter(w2, requires_grad=True)

    def encode(self, f, m_warped):
        diff = f - m_warped
        h1 = self.conv1(diff)
        return self.conv2(h1)

    def forward(self, f, m_warped):
        return self.encode(f, m_warped)


class SimEnc(BaseModel):
    def __init__(self, s):
        super(SimEnc, self).__init__()
        self.CNN = CNN(s)

    def forward(self, f, m_warped):
        return self.CNN.forward(f, m_warped)

