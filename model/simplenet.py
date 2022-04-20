import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(SimpleNet, self).__init__()

        self.conv_layer1 = self._basic_block(in_channels, 32)
        self.conv_layer2 = self._basic_block(32, 64)
        self.conv_layer3 = self._basic_block(64, 128)

        self.drop = nn.Dropout(0.5)
        self.classify_block = self._classify_block(128, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _basic_block(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        return conv_layer

    def _classify_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.drop(x)
        x = self.classify_block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def generate_model(use_gpu=True, gpu_id=['1'], in_channels=3, num_classes=10):
    model = SimpleNet(in_channels, num_classes)

    if use_gpu:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id[0])
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)

    return model


if __name__ == '__main__':
    model = generate_model(use_gpu=False)
    x = torch.rand(4, 64, 64, 3)
    out = model(x)
    print(out)
