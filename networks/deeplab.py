import torch.nn as nn
import torch.nn.functional as F
affine_par = True

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

class ResNet_HDL(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet_HDL, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilation=4)
        if block.expansion == 4:
            self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [
                                            6, 12, 18, 24], num_classes)
        else:
            self.layer5 = self._make_pred_layer(Classifier_Module, 512, [6, 12, 18, 24], [
                                            6, 12, 18, 24], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
            for i in downsample._modules['1'].parameters():
                i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)
    def forward(self, x, feat_=None, src=None, lambda_trg=None):
        _, _, h, w = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if src == 0:
            if feat_ is None:  # Target extracting
                x1 = self.layer1(x)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
                x5 = self.layer5(x4)
                #fa = [f1.detach(), f2.detach(), f3.detach(), f4.detach()]   # 分别提取几层源域的se向量
            # x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
                return [x1, x2, x3, x4, x5]
            else:   # target training
                x1 = self.layer1(x)
                x2 = self.layer2(x1)
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
                x5 = self.layer5(x4)
                return x5
        elif src == 1:   # Source training
            x1 = self.layer1(x)
            b, c, h, w = x1.shape
            # mask1 = torch.rand(b, 1, h, w).cuda()
            # #print("mask1", h ,' ', w)
            # mask1 = torch.where(mask1>0.5, 1, 0)
            feat_1 = F.interpolate(feat_[0], size=(h, w), mode='bilinear', align_corners=True)
            x1_mix = (1-lambda_trg) * x1 + lambda_trg * feat_1

            x2 = self.layer2(x1_mix)
            b, c, h, w = x2.shape
            # mask2 = torch.rand(b, 1, h, w).cuda()
            # mask2 = torch.where(mask2 > 0.5, 1, 0)
            feat_2 = F.interpolate(feat_[1], size=(h, w), mode='bilinear', align_corners=True)
            x2_mix = (1-lambda_trg) * x2 + lambda_trg * feat_2
            #print("mask2", h ,' ', w)

            x3 = self.layer3(x2_mix)
            b, c, h, w = x3.shape
            # mask3 = torch.rand(b, 1, h, w).cuda()
            # mask3 = torch.where(mask3 > 0.5, 1, 0)
            feat_3 = F.interpolate(feat_[2], size=(h, w), mode='bilinear', align_corners=True)
            #print("mask3", h ,' ', w)

            x3_mix = (1 - lambda_trg) * x3 + lambda_trg * feat_3

            x4 = self.layer4(x3_mix)
            # b, c, h, w = x4.shape
            # mask4 = torch.rand(b, 1, h, w)
            # mask4 = torch.where(mask4 > 0.5, 1, 0)
            # x4_mix = (1 - lambda_trg) * x4 + lambda_trg * feat_[3] * mask4

            x5 = self.layer5(x4)
            #print('mask num :', mask1.sum(), ' ', mask2.sum(), ' ', mask3.sum())
            return x5
        elif src == 2:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            return x
    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k
    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i
    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]

def Deeplab_Res101HDL(num_classes=21):
    print("Deeplab_Res101_HDL... ")
    model = ResNet_HDL(Bottleneck, [3, 4, 23, 3], num_classes)
    return model

