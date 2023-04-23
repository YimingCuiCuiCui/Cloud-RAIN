import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from utils.dgcnn_util import get_graph_feature, Transform_Net

class QDGCNN_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(QDGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1_r = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv1_g = nn.Conv2d(6, 64, kernel_size=1, bias=True)
        self.conv1_b = nn.Conv2d(6, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv2_r = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv2_g = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=True)
        self.conv2_b = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv2_g.weight.data.fill_(0)
        self.conv2_g.bias.data.fill_(1)
        self.conv2_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv3_r = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv3_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        # self.conv3_b.bias.data.fill_(0)

        self.conv4_r = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv4_g = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=True)
        self.conv4_b = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv4_r.weight)
        self.conv4_g.weight.data.fill_(0)
        self.conv4_g.bias.data.fill_(1)
        self.conv4_b.weight.data.fill_(0)
        # self.conv4_b.bias.data.fill_(0)

        self.conv5_r = nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False)
        self.conv5_g = nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=True)
        self.conv5_b = nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv5_r.weight)
        self.conv5_g.weight.data.fill_(0)
        self.conv5_g.bias.data.fill_(1)
        self.conv5_b.weight.data.fill_(0)
        # self.conv5_b.bias.data.fill_(0)

        self.active_func1 = nn.Sequential(self.bn1,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func2 = nn.Sequential(self.bn2,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func3 = nn.Sequential(self.bn3,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func4 = nn.Sequential(self.bn4,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func5 = nn.Sequential(self.bn5,
                                          nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.active_func1(self.conv1_r(x) * self.conv1_g(x) + self.conv1_b(x.mul(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func2(self.conv2_r(x) * self.conv2_g(x) + self.conv2_b(x.mul(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func3(self.conv3_r(x) * self.conv3_g(x) + self.conv3_b(x.mul(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.active_func4(self.conv4_r(x) * self.conv4_g(x) + self.conv4_b(x.mul(x)))
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        x = self.active_func5(self.conv5_r(x) * self.conv5_g(x) + self.conv5_b(x.mul(x)))
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size,
                                              -1)  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)  # (batch_size, emb_dims*2)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)  # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        x = self.linear3(x)  # (batch_size, 256) -> (batch_size, output_channels)

        return x

class QDGCNN_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(QDGCNN_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        self.conv1_r = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv1_g = nn.Conv2d(6, 64, kernel_size=1, bias=True)
        self.conv1_b = nn.Conv2d(6, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv2_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv2_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv2_g.weight.data.fill_(0)
        self.conv2_g.bias.data.fill_(1)
        self.conv2_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv3_r = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv3_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        # self.conv3_b.bias.data.fill_(0)

        self.conv4_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv4_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv4_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv4_g.weight.data.fill_(0)
        self.conv4_g.bias.data.fill_(1)
        self.conv4_b.weight.data.fill_(0)
        # self.conv4_b.bias.data.fill_(0)

        self.conv5_r = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv5_g = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.conv5_b = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv5_g.weight.data.fill_(0)
        self.conv5_g.bias.data.fill_(1)
        self.conv5_b.weight.data.fill_(0)

        self.conv6_r = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False)
        self.conv6_g = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=True)
        self.conv6_b = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv6_g.weight.data.fill_(0)
        self.conv6_g.bias.data.fill_(1)
        self.conv6_b.weight.data.fill_(0)
        # self.conv6_b.bias.data.fill_(0)

        self.conv7_r = nn.Conv1d(16, 64, kernel_size=1, bias=False)
        self.conv7_g = nn.Conv1d(16, 64, kernel_size=1, bias=True)
        self.conv7_b = nn.Conv1d(16, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv7_g.weight.data.fill_(0)
        self.conv7_g.bias.data.fill_(1)
        self.conv7_b.weight.data.fill_(0)
        # self.conv7_b.bias.data.fill_(0)

        self.conv8_r = nn.Conv1d(1280, 256, kernel_size=1, bias=False)
        self.conv8_g = nn.Conv1d(1280, 256, kernel_size=1, bias=True)
        self.conv8_b = nn.Conv1d(1280, 256, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv8_g.weight.data.fill_(0)
        self.conv8_g.bias.data.fill_(1)
        self.conv8_b.weight.data.fill_(0)
        # self.conv8_b.bias.data.fill_(0)

        self.active_func1 = nn.Sequential(self.bn1,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func2 = nn.Sequential(self.bn2,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func3 = nn.Sequential(self.bn3,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func4 = nn.Sequential(self.bn4,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func5 = nn.Sequential(self.bn5,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func6 = nn.Sequential(self.bn6,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func7 = nn.Sequential(self.bn7,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func8 = nn.Sequential(self.bn8,
                                          nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.active_func1(self.conv1_r(x) * self.conv1_g(x) + self.conv1_b(x.mul(x)))
        x = self.active_func2(self.conv2_r(x) * self.conv2_g(x) + self.conv2_b(x.mul(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func3(self.conv3_r(x) * self.conv3_g(x) + self.conv3_b(x.mul(x)))
        x = self.active_func4(self.conv4_r(x) * self.conv4_g(x) + self.conv4_b(x.mul(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func5(self.conv5_r(x) * self.conv5_g(x) + self.conv5_b(x.mul(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.active_func6(self.conv6_r(x) * self.conv6_g(x) + self.conv6_b(x.mul(x)))
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        l = self.active_func7(self.conv7_r(l) * self.conv7_g(l) + self.conv7_b(l.mul(l)))

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        x = self.active_func8(self.conv8_r(x) * self.conv8_g(x) + self.conv8_b(x.mul(x)))
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x

class QDGCNN_semseg_s3dis2(nn.Module):
    def __init__(self, args):
        super(QDGCNN_semseg_s3dis2, self).__init__()
        self.args = args
        self.k = args.k
        self.bn1 = nn.BatchNorm2d(1024)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv1_r = nn.Conv2d(18, 1024, kernel_size=1, bias=False)
        self.conv1_g = nn.Conv2d(18, 1024, kernel_size=1, bias=True)
        self.conv1_b = nn.Conv2d(18, 1024, kernel_size=1, bias=True)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        self.conv1_b.bias.data.fill_(0)

        self.active_func1 = nn.Sequential(self.bn1,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.active_func1(self.conv1_r(x) * self.conv1_g(x) + self.conv1_b(x.mul(x)))
        x = x.max(dim=-1, keepdim=False)[0]  
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.repeat(1, 1, num_points)     
        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x

class QDGCNN_semseg_s3dis(nn.Module):
    def __init__(self, args):
        super(QDGCNN_semseg_s3dis, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv1_r = nn.Conv2d(18, 64, kernel_size=1, bias=False)
        self.conv1_g = nn.Conv2d(18, 64, kernel_size=1, bias=True)
        self.conv1_b = nn.Conv2d(18, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv2_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv2_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv2_g.weight.data.fill_(0)
        self.conv2_g.bias.data.fill_(1)
        self.conv2_b.weight.data.fill_(0)
        #self.conv2_b.bias.data.fill_(0)

        self.conv3_r = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv3_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        #self.conv3_b.bias.data.fill_(0)

        self.conv4_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv4_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv4_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv4_g.weight.data.fill_(0)
        self.conv4_g.bias.data.fill_(1)
        self.conv4_b.weight.data.fill_(0)
        #self.conv4_b.bias.data.fill_(0)

        self.conv5_r = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv5_g = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.conv5_b = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv5_g.weight.data.fill_(0)
        self.conv5_g.bias.data.fill_(1)
        self.conv5_b.weight.data.fill_(0)
        #self.conv5_b.bias.data.fill_(0)

        self.conv6_r = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False)
        self.conv6_g = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=True)
        self.conv6_b = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv6_g.weight.data.fill_(0)
        self.conv6_g.bias.data.fill_(1)
        self.conv6_b.weight.data.fill_(0)
        #self.conv6_b.bias.data.fill_(0)

        self.conv7_r = nn.Conv1d(1216, 512, kernel_size=1, bias=False)
        self.conv7_g = nn.Conv1d(1216, 512, kernel_size=1, bias=True)
        self.conv7_b = nn.Conv1d(1216, 512, kernel_size=1, bias=False)

        ## init.kaiming_normal_(self.conv1_r.weight)
        self.conv7_g.weight.data.fill_(0)
        self.conv7_g.bias.data.fill_(1)
        self.conv7_b.weight.data.fill_(0)
        #self.conv7_b.bias.data.fill_(0)

        self.conv8_r = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv8_g = nn.Conv1d(512, 256, kernel_size=1, bias=True)
        self.conv8_b = nn.Conv1d(512, 256, kernel_size=1, bias=False)

        ## init.kaiming_normal_(self.conv1_r.weight)
        self.conv8_g.weight.data.fill_(0)
        self.conv8_g.bias.data.fill_(1)
        self.conv8_b.weight.data.fill_(0)
        ## self.conv8_b.bias.data.fill_(0)

        self.active_func1 = nn.Sequential(self.bn1,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func2 = nn.Sequential(self.bn2,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func3 = nn.Sequential(self.bn3,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func4 = nn.Sequential(self.bn4,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func5 = nn.Sequential(self.bn5,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func6 = nn.Sequential(self.bn6,
                                          nn.LeakyReLU(negative_slope=0.2))
        # self.active_func7 = nn.Sequential(self.bn7,
        #                                   nn.LeakyReLU(negative_slope=0.2))
        # self.active_func8 = nn.Sequential(self.bn8,
        #                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.active_func1(self.conv1_r(x) * self.conv1_g(x) + self.conv1_b(x.mul(x)))
        x = self.active_func2(self.conv2_r(x) * self.conv2_g(x) + self.conv2_b(x.mul(x)))
#        x = self.active_func1(self.conv1_b(x.mul(x)))
#        x = self.active_func2(self.conv2_b(x.mul(x))) 
        
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func3(self.conv3_r(x) * self.conv3_g(x) + self.conv3_b(x.mul(x)))
        x = self.active_func4(self.conv4_r(x) * self.conv4_g(x) + self.conv4_b(x.mul(x)))
#        x = self.active_func3(self.conv3_b(x.mul(x)))
#        x = self.active_func4(self.conv4_b(x.mul(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func5(self.conv5_r(x) * self.conv5_g(x) + self.conv5_b(x.mul(x)))
#        x = self.active_func5(self.conv5_b(x.mul(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

#        x = self.active_func6(self.conv6_b(x.mul(x)))
        x = self.active_func6(self.conv6_r(x) * self.conv6_g(x) + self.conv6_b(x.mul(x)))
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x

class QDGCNN_semseg_scannet(nn.Module):
    def __init__(self, args):
        super(QDGCNN_semseg_scannet, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv1_r = nn.Conv2d(12, 64, kernel_size=1, bias=False)
        self.conv1_g = nn.Conv2d(12, 64, kernel_size=1, bias=True)
        self.conv1_b = nn.Conv2d(12, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv2_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv2_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv2_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv2_g.weight.data.fill_(0)
        self.conv2_g.bias.data.fill_(1)
        self.conv2_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv3_r = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv3_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        # self.conv3_b.bias.data.fill_(0)

        self.conv4_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv4_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv4_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv4_g.weight.data.fill_(0)
        self.conv4_g.bias.data.fill_(1)
        self.conv4_b.weight.data.fill_(0)
        # self.conv4_b.bias.data.fill_(0)

        self.conv5_r = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv5_g = nn.Conv2d(64*2, 64, kernel_size=1, bias=True)
        self.conv5_b = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv5_g.weight.data.fill_(0)
        self.conv5_g.bias.data.fill_(1)
        self.conv5_b.weight.data.fill_(0)
        # self.conv5_b.bias.data.fill_(0)

        self.conv6_r = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False)
        self.conv6_g = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=True)
        self.conv6_b = nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv6_g.weight.data.fill_(0)
        self.conv6_g.bias.data.fill_(1)
        self.conv6_b.weight.data.fill_(0)
        # self.conv6_b.bias.data.fill_(0)

        # self.conv7_r = nn.Conv1d(1216, 512, kernel_size=1, bias=False)
        # self.conv7_g = nn.Conv1d(1216, 512, kernel_size=1, bias=True)
        # self.conv7_b = nn.Conv1d(1216, 512, kernel_size=1, bias=False)

        ## init.kaiming_normal_(self.conv1_r.weight)
        # self.conv7_g.weight.data.fill_(0)
        # self.conv7_g.bias.data.fill_(1)
        # self.conv7_b.weight.data.fill_(0)
        ## self.conv7_b.bias.data.fill_(0)

        # self.conv8_r = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        # self.conv8_g = nn.Conv1d(512, 256, kernel_size=1, bias=True)
        # self.conv8_b = nn.Conv1d(512, 256, kernel_size=1, bias=False)

        ## init.kaiming_normal_(self.conv1_r.weight)
        # self.conv8_g.weight.data.fill_(0)
        # self.conv8_g.bias.data.fill_(1)
        # self.conv8_b.weight.data.fill_(0)
        ## self.conv8_b.bias.data.fill_(0)

        self.active_func1 = nn.Sequential(self.bn1,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func2 = nn.Sequential(self.bn2,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func3 = nn.Sequential(self.bn3,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func4 = nn.Sequential(self.bn4,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func5 = nn.Sequential(self.bn5,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func6 = nn.Sequential(self.bn6,
                                          nn.LeakyReLU(negative_slope=0.2))
        # self.active_func7 = nn.Sequential(self.bn7,
        #                                   nn.LeakyReLU(negative_slope=0.2))
        # self.active_func8 = nn.Sequential(self.bn8,
        #                                   nn.LeakyReLU(negative_slope=0.2))

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 21, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.active_func1(self.conv1_r(x) * self.conv1_g(x) + self.conv1_b(x.mul(x)))
        x = self.active_func2(self.conv2_r(x) * self.conv2_g(x) + self.conv2_b(x.mul(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func3(self.conv3_r(x) * self.conv3_g(x) + self.conv3_b(x.mul(x)))
        x = self.active_func4(self.conv4_r(x) * self.conv4_g(x) + self.conv4_b(x.mul(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func5(self.conv5_r(x) * self.conv5_g(x) + self.conv5_b(x.mul(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.active_func6(self.conv6_r(x) * self.conv6_g(x) + self.conv6_b(x.mul(x)))
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x
