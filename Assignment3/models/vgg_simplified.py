import torch
import torch.nn as nn
import math

class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)
        # for all conv layers, set: kernel=3, padding=1

        self.conv_block1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                                   out_channels=64,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))

        self.conv_block2 = nn.Sequential(nn.Conv2d(in_channels=64,
                                                   out_channels=128,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))

        self.conv_block3 = nn.Sequential(nn.Conv2d(in_channels=128,
                                                   out_channels=256,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))

        self.conv_block4 = nn.Sequential(nn.Conv2d(in_channels=256,
                                                   out_channels=512,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))

        self.conv_block5 = nn.Sequential(nn.Conv2d(in_channels=512,
                                                   out_channels=512,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2,
                                                      stride=2))

        self.final_classifier = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                              nn.ReLU(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(in_features=256, out_features=10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        score = None

        ris = x.clone()
        s_stacked = nn.Sequential(self.conv_block1,
                                  self.conv_block2,
                                  self.conv_block3,
                                  self.conv_block4,
                                  self.conv_block5)
        ris = s_stacked(ris)
        ris = ris[:,:,0,0] # The input to the linear layer must be a 2D tensor
        score = self.final_classifier(ris)

        return score

