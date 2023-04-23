import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from utils.pointnet_util import QPointNetEncoder, feature_transform_reguliarzer, STNkd, STN3d

class QPointNet_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(QPointNet_cls, self).__init__()

        self.feat = QPointNetEncoder(global_feat=True, feature_transform=True, channel=3)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_channels)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        regular_loss = 0.001 * feature_transform_reguliarzer(trans_feat)
        return x, regular_loss

class QPointNet_partseg(nn.Module):
    def __init__(self, args, part_num=50, normal_channel=False):
        super(QPointNet_partseg, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)

        self.conv1_r = torch.nn.Conv1d(channel, 64, 1)
        self.conv1_g = torch.nn.Conv1d(channel, 64, 1)
        self.conv1_b = torch.nn.Conv1d(channel, 64, 1)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        self.conv1_b.bias.data.fill_(0)

        self.conv2_r = torch.nn.Conv1d(64, 128, 1)
        self.conv2_g = torch.nn.Conv1d(64, 128, 1)
        self.conv2_b = torch.nn.Conv1d(64, 128, 1)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv2_g.weight.data.fill_(0)
        self.conv2_g.bias.data.fill_(1)
        self.conv2_b.weight.data.fill_(0)
        self.conv2_b.bias.data.fill_(0)

        self.conv3_r = torch.nn.Conv1d(128, 128, 1)
        self.conv3_g = torch.nn.Conv1d(128, 128, 1)
        self.conv3_b = torch.nn.Conv1d(128, 128, 1)

        # init.kaiming_normal_(self.conv3_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        self.conv3_b.bias.data.fill_(0)

        self.conv4_r = torch.nn.Conv1d(128, 512, 1)
        self.conv4_g = torch.nn.Conv1d(128, 512, 1)
        self.conv4_b = torch.nn.Conv1d(128, 512, 1)

        # init.kaiming_normal_(self.conv4_r.weight)
        self.conv4_g.weight.data.fill_(0)
        self.conv4_g.bias.data.fill_(1)
        self.conv4_b.weight.data.fill_(0)
        self.conv4_b.bias.data.fill_(0)

        self.conv5_r = torch.nn.Conv1d(512, 2048, 1)
        self.conv5_g = torch.nn.Conv1d(512, 2048, 1)
        self.conv5_b = torch.nn.Conv1d(512, 2048, 1)

        # init.kaiming_normal_(self.conv5_r.weight)
        self.conv5_g.weight.data.fill_(0)
        self.conv5_g.bias.data.fill_(1)
        self.conv5_b.weight.data.fill_(0)
        self.conv5_b.bias.data.fill_(0)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1_r(point_cloud) * self.conv1_g(point_cloud) + self.conv1_b(point_cloud.mul(point_cloud))))
        out2 = F.relu(self.bn2(self.conv2_r(out1) * self.conv2_g(out1) + self.conv2_b(out1.mul(out1))))
        out3 = F.relu(self.bn3(self.conv3_r(out2) * self.conv3_g(out2) + self.conv3_b(out2.mul(out2))))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4_r(net_transformed) * self.conv4_g(net_transformed) + self.conv4_b(net_transformed.mul(net_transformed))))
        out5 = self.bn5(self.conv5_r(out4) * self.conv5_g(out4) + self.conv5_b(out4.mul(out4)))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net)
        # net = net.transpose(2, 1).contiguous()
        # net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, self.part_num, N) # [B, N, 50]

        regular_loss = 0.001 * feature_transform_reguliarzer(trans_feat)
        return net, regular_loss.mean()



class QPointNet_semseg_s3dis(nn.Module):
    def __init__(self, args, num_class=13):
        super(QPointNet_semseg_s3dis, self).__init__()
        self.k = num_class
        self.feat = QPointNetEncoder(global_feat=False, feature_transform=True, channel=9)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, self.k, n_pts)
        regular_loss = 0.001 * feature_transform_reguliarzer(trans_feat)
        return x, regular_loss.sum()

class QPointNet_semseg_scannet(nn.Module):
    def __init__(self, args, num_class=21):
        super(QPointNet_semseg_scannet, self).__init__()
        self.k = num_class
        self.feat = QPointNetEncoder(global_feat=False, feature_transform=True, channel=6)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, self.k, n_pts)
        regular_loss = 0.001 * feature_transform_reguliarzer(trans_feat)
        return x, regular_loss.sum()
