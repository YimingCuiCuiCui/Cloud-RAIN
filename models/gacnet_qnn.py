import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from utils.gacnet_util import QGraphAttentionConvLayer, QPointNetFeaturePropagation


class QGACNet_semseg_s3dis(nn.Module):
    def __init__(self, num_classes=13,dropout=0.5,alpha=0.2):
        super(QGACNet_semseg_s3dis, self).__init__()
        # GraphAttentionConvLayer: npoint, radius, nsample, in_channel, mlp, group_all,dropout,alpha
        self.sa1 = QGraphAttentionConvLayer(1024, 0.1, 32, 6 + 3, [32, 64], False, dropout,alpha)
        self.sa2 = QGraphAttentionConvLayer(256, 0.2, 32, 64 + 3, [64, 128], False, dropout,alpha)
        self.sa3 = QGraphAttentionConvLayer(64, 0.4, 32, 128 + 3, [128, 256], False, dropout,alpha)
        self.sa4 = QGraphAttentionConvLayer(16, 0.8, 32, 256 + 3, [256, 512], False, dropout,alpha)
        # PointNetFeaturePropagation: in_channel, mlp
        self.fp4 = QPointNetFeaturePropagation(768, [256, 256])
        self.fp3 = QPointNetFeaturePropagation(384, [256, 256])
        self.fp2 = QPointNetFeaturePropagation(320, [256, 128])
        self.fp1 = QPointNetFeaturePropagation(128, [128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        xyz, point = x[:,:3,:], x[:,3:,:]
        l1_xyz, l1_points = self.sa1(xyz, point)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x


class QGACNet_semseg_scannet(nn.Module):
    def __init__(self, num_classes=21,dropout=0.5,alpha=0.2):
        super(QGACNet_semseg_scannet, self).__init__()
        # GraphAttentionConvLayer: npoint, radius, nsample, in_channel, mlp, group_all,dropout,alpha
        self.sa1 = QGraphAttentionConvLayer(1024, 0.1, 32, 3 + 3, [32, 64], False, dropout,alpha)
        self.sa2 = QGraphAttentionConvLayer(256, 0.2, 32, 64 + 3, [64, 128], False, dropout,alpha)
        self.sa3 = QGraphAttentionConvLayer(64, 0.4, 32, 128 + 3, [128, 256], False, dropout,alpha)
        self.sa4 = QGraphAttentionConvLayer(16, 0.8, 32, 256 + 3, [256, 512], False, dropout,alpha)
        # PointNetFeaturePropagation: in_channel, mlp
        self.fp4 = QPointNetFeaturePropagation(768, [256, 256])
        self.fp3 = QPointNetFeaturePropagation(384, [256, 256])
        self.fp2 = QPointNetFeaturePropagation(320, [256, 128])
        self.fp1 = QPointNetFeaturePropagation(128, [128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        xyz, point = x[:,:3,:], x[:,3:,:]
        l1_xyz, l1_points = self.sa1(xyz, point)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # x = F.log_softmax(x, dim=1)
        # x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,9,2048))
    model = QGACNet_semseg_s3dis(13)
    output = model(input)
    print(input.shape, output.shape)
    input = torch.randn((8,6,2048))
    model = QGACNet_semseg_scannet(21)
    output = model(input)
    print(input.shape, output.shape)
