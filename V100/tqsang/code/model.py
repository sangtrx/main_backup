import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# class OCR(nn.Module):
#     def __init__(self, vocab_size):
#         super(OCR, self).__init__()
#         # self.conv = models.vgg16_bn(pretrained=True).features
#         # self.conv[23] = nn.MaxPool2d((2, 1), (2, 1))
#         # self.conv[33] = nn.MaxPool2d((2, 1), (2, 1))
#         # self.conv[43] = nn.MaxPool2d((2, 1), (2, 1))
#         # self.rnn = nn.LSTM(input_size=512, hidden_size=256, bidirectional=False)
#         # self.out = nn.Linear(256, vocab_size)

#         self.conv = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-4])
#         self.rnn = nn.LSTM(input_size=512, hidden_size=256, bidirectional=False)
#         self.out = nn.Linear(256, vocab_size)

#     def forward(self, images, targets=None):
#         images = self.conv(images)  # B, C, H, W
#         images = torch.mean(images, dim=2, keepdim=False)  # B, C, W
#         images = images.permute(2, 0, 1)  # T=W, B, C
#         images, _ = self.rnn(images)
#         images = self.out(images)  # T, B, V
#         if self.training:
#             assert targets is not None
#             images = F.log_softmax(images, dim=-1)
#         else:
#             images = F.softmax(images, dim=-1)
#             images = images.transpose(0, 1)
#         return images


class OCR(nn.Module):
    def __init__(self, vocab_size):
        super(OCR, self).__init__()
        # self.conv = models.vgg16_bn(pretrained=True).features
        # self.conv[6] = nn.MaxPool2d((2, 1), (2, 1))
        # self.conv[13] = nn.MaxPool2d((2, 1), (2, 1))
        # self.conv[23] = nn.MaxPool2d((2, 1), (2, 1))
        # self.conv[33] = nn.MaxPool2d((2, 1), (2, 1))
        # self.conv[43] = nn.MaxPool2d((2, 1), (2, 1))
        self.out = nn.Linear(512, vocab_size)
        resnet = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2])
        resnet[0].kernel_size = (3, 3)
        resnet[0].padding = (1, 1)
        resnet[0].stride = (1, 1)
        resnet[3] = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        # resnet[-1][0].conv1.stride = (2, 1)
        # resnet[-1][0].downsample[0].stride = (2, 1)
        resnet[-2][0].conv1.stride = (2, 1)
        resnet[-2][0].downsample[0].stride = (2, 1)
        resnet[-3][0].conv1.stride = (2, 1)
        resnet[-3][0].downsample[0].stride = (2, 1)
        self.conv = resnet
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)
        # self.out = nn.Linear(256, vocab_size)

    def forward(self, images, targets=None):
        images = self.conv(images)  # B, C, H, W
        # images = torch.max(images, dim=2, keepdim=False)[0]  # B, C, W
        images = torch.mean(images, dim=2, keepdim=False)  # B, C, W
        images = images.permute(2, 0, 1)  # T=W, B, C
        images, _ = self.rnn(torch.flipud(images))
        images = self.out(images)  # T, B, V
        if self.training:
            assert targets is not None
            images = F.log_softmax(images, dim=-1)
        else:
            images = F.softmax(images, dim=-1)
            images = images.transpose(0, 1)

        return images
