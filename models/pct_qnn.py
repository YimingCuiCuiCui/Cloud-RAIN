import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")

from utils.pct_util import QSA_Layer

class QPCT_semseg_s3dis(nn.Module):
    def __init__(self, args):
        super(QPCT_semseg_s3dis, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(9, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = QSA_Layer(128)
        self.sa2 = QSA_Layer(128)
        self.sa3 = QSA_Layer(128)
        self.sa4 = QSA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 13, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)

        x_max = torch.max(x, 2, keepdim=True)[0]
        x_avg = torch.mean(x, 2, keepdim=True)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1024 + 64

        x = torch.cat((x, x_global_feature), 1)  # 1024 * 2 + 64

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x

class QPCT_semseg_scannet(nn.Module):
    def __init__(self, args):
        super(QPCT_semseg_scannet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(6, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

        self.sa1 = QSA_Layer(128)
        self.sa2 = QSA_Layer(128)
        self.sa3 = QSA_Layer(128)
        self.sa4 = QSA_Layer(128)

        self.conv_fuse = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(1024),
                                       nn.LeakyReLU(negative_slope=0.2))

        self.convs1 = nn.Conv1d(1024 * 3, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, 21, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)

        x_max = torch.max(x, 2, keepdim=True)[0]
        x_avg = torch.mean(x, 2, keepdim=True)
        x_max_feature = x_max.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)
        x_avg_feature = x_avg.view(batch_size, -1).unsqueeze(-1).repeat(1, 1, N)

        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)  # 1024 + 64

        x = torch.cat((x, x_global_feature), 1)  # 1024 * 2 + 64

        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        return x

if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    args = parser.parse_args()
    args.k = 20
    args.emb_dims = 1024
    args.dropout = 0.5
    print(args)

    input = torch.randn((8,9,2048)).cuda()
    model = QPCT_semseg_s3dis(args).cuda()
    output = model(input)
    print(input.shape, output.shape, sum(p.numel() for p in model.parameters()))
    input = torch.randn((8,6,2048)).cuda()
    model = QPCT_semseg_scannet(args).cuda()
    output = model(input)
    print(input.shape, output.shape, sum(p.numel() for p in model.parameters()))
