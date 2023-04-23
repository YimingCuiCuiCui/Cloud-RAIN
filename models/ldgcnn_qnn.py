import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
from utils.dgcnn_util import get_graph_feature, Transform_Net

class QLDGCNN_semseg_s3dis(nn.Module):
    def __init__(self, args):
        super(QLDGCNN_semseg_s3dis, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        # self.bn7 = nn.BatchNorm2d(128)
        # self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.bn10 = nn.BatchNorm1d(512)
        self.bn11 = nn.BatchNorm1d(256)

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
        # self.conv2_b.bias.data.fill_(0)

        self.conv3_r = nn.Conv2d(73 * 2, 64, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(73 * 2, 64, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(73 * 2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv4_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv4_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv4_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv4_g.weight.data.fill_(0)
        self.conv4_g.bias.data.fill_(1)
        self.conv4_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv5_r = nn.Conv2d(137 * 2, 64, kernel_size=1, bias=False)
        self.conv5_g = nn.Conv2d(137 * 2, 64, kernel_size=1, bias=True)
        self.conv5_b = nn.Conv2d(137 * 2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv5_g.weight.data.fill_(0)
        self.conv5_g.bias.data.fill_(1)
        self.conv5_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv6_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv6_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv6_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv6_g.weight.data.fill_(0)
        self.conv6_g.bias.data.fill_(1)
        self.conv6_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        # self.conv7_r = nn.Conv2d(201 * 2, 128, kernel_size=1, bias=False)
        # self.conv7_g = nn.Conv2d(201 * 2, 128, kernel_size=1, bias=True)
        # self.conv7_b = nn.Conv2d(201 * 2, 128, kernel_size=1, bias=False)
        #
        # # init.kaiming_normal_(self.conv1_r.weight)
        # self.conv7_g.weight.data.fill_(0)
        # self.conv7_g.bias.data.fill_(1)
        # self.conv7_b.weight.data.fill_(0)
        # # self.conv1_b.bias.data.fill_(0)
        #
        # self.conv8_r = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        # self.conv8_g = nn.Conv2d(128, 128, kernel_size=1, bias=True)
        # self.conv8_b = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        #
        # # init.kaiming_normal_(self.conv2_r.weight)
        # self.conv8_g.weight.data.fill_(0)
        # self.conv8_g.bias.data.fill_(1)
        # self.conv8_b.weight.data.fill_(0)
        # # self.conv2_b.bias.data.fill_(0)

        self.conv9_r = nn.Conv1d(201, args.emb_dims, kernel_size=1, bias=False)
        self.conv9_g = nn.Conv1d(201, args.emb_dims, kernel_size=1, bias=True)
        self.conv9_b = nn.Conv1d(201, args.emb_dims, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv9_g.weight.data.fill_(0)
        self.conv9_g.bias.data.fill_(1)
        self.conv9_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

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
        self.active_func9 = nn.Sequential(self.bn9,
                                          nn.LeakyReLU(negative_slope=0.2))

        self.conv10 = nn.Sequential(nn.Conv1d(args.emb_dims + 192, 512, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn11,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv12 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = x
        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.active_func1(self.conv1_r(x) * self.conv1_g(x) + self.conv1_b(x.mul(x)))
        x = self.active_func2(self.conv2_r(x) * self.conv2_g(x) + self.conv2_b(x.mul(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(torch.cat((x0, x1), dim=1), k=self.k)  # (batch_size, 73, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func3(self.conv3_r(x) * self.conv3_g(x) + self.conv3_b(x.mul(x)))
        x = self.active_func4(self.conv4_r(x) * self.conv4_g(x) + self.conv4_b(x.mul(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(torch.cat((x0, x1, x2), dim=1), k=self.k)  # (batch_size, 137, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func5(self.conv5_r(x) * self.conv5_g(x) + self.conv5_b(x.mul(x)))
        x = self.active_func6(self.conv6_r(x) * self.conv6_g(x) + self.conv6_b(x.mul(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # x = get_graph_feature(torch.cat((x0, x1, x2, x3), dim=1), k=self.k)  # (batch_size, 137, num_points) -> (batch_size, 64*2, num_points, k)
        # x = self.active_func7(self.conv7_r(x) * self.conv7_g(x) + self.conv7_b(x.mul(x)))
        # x = self.active_func8(self.conv8_r(x) * self.conv8_g(x) + self.conv8_b(x.mul(x)))
        # x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x0, x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.active_func9(self.conv9_r(x) * self.conv9_g(x) + self.conv9_b(x.mul(x)))
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv10(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv11(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv12(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        return x

class QLDGCNN_semseg_scannet(nn.Module):
    def __init__(self, args):
        super(QLDGCNN_semseg_scannet, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        # self.bn7 = nn.BatchNorm2d(128)
        # self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm1d(args.emb_dims)
        self.bn10 = nn.BatchNorm1d(512)
        self.bn11 = nn.BatchNorm1d(256)

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

        self.conv3_r = nn.Conv2d(70 * 2, 64, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(70 * 2, 64, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(70 * 2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv4_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv4_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv4_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv4_g.weight.data.fill_(0)
        self.conv4_g.bias.data.fill_(1)
        self.conv4_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv5_r = nn.Conv2d(134 * 2, 64, kernel_size=1, bias=False)
        self.conv5_g = nn.Conv2d(134 * 2, 64, kernel_size=1, bias=True)
        self.conv5_b = nn.Conv2d(134 * 2, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv5_g.weight.data.fill_(0)
        self.conv5_g.bias.data.fill_(1)
        self.conv5_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv6_r = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv6_g = nn.Conv2d(64, 64, kernel_size=1, bias=True)
        self.conv6_b = nn.Conv2d(64, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv6_g.weight.data.fill_(0)
        self.conv6_g.bias.data.fill_(1)
        self.conv6_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv7_r = nn.Conv2d(198 * 2, 128, kernel_size=1, bias=False)
        self.conv7_g = nn.Conv2d(198 * 2, 128, kernel_size=1, bias=True)
        self.conv7_b = nn.Conv2d(198 * 2, 128, kernel_size=1, bias=False)

        # # init.kaiming_normal_(self.conv1_r.weight)
        # self.conv7_g.weight.data.fill_(0)
        # self.conv7_g.bias.data.fill_(1)
        # self.conv7_b.weight.data.fill_(0)
        # # self.conv1_b.bias.data.fill_(0)
        #
        # self.conv8_r = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        # self.conv8_g = nn.Conv2d(128, 128, kernel_size=1, bias=True)
        # self.conv8_b = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        #
        # # init.kaiming_normal_(self.conv2_r.weight)
        # self.conv8_g.weight.data.fill_(0)
        # self.conv8_g.bias.data.fill_(1)
        # self.conv8_b.weight.data.fill_(0)
        # # self.conv2_b.bias.data.fill_(0)

        self.conv9_r = nn.Conv1d(198, args.emb_dims, kernel_size=1, bias=False)
        self.conv9_g = nn.Conv1d(198, args.emb_dims, kernel_size=1, bias=True)
        self.conv9_b = nn.Conv1d(198, args.emb_dims, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv9_g.weight.data.fill_(0)
        self.conv9_g.bias.data.fill_(1)
        self.conv9_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

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
        self.active_func9 = nn.Sequential(self.bn9,
                                          nn.LeakyReLU(negative_slope=0.2))

        self.conv10 = nn.Sequential(nn.Conv1d(args.emb_dims + 192, 512, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                    self.bn11,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv12 = nn.Conv1d(256, 21, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = x
        x = get_graph_feature(x, k=self.k, dim9=True)  # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.active_func1(self.conv1_r(x) * self.conv1_g(x) + self.conv1_b(x.mul(x)))
        x = self.active_func2(self.conv2_r(x) * self.conv2_g(x) + self.conv2_b(x.mul(x)))
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(torch.cat((x0, x1), dim=1),
                              k=self.k)  # (batch_size, 73, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func3(self.conv3_r(x) * self.conv3_g(x) + self.conv3_b(x.mul(x)))
        x = self.active_func4(self.conv4_r(x) * self.conv4_g(x) + self.conv4_b(x.mul(x)))
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(torch.cat((x0, x1, x2), dim=1),
                              k=self.k)  # (batch_size, 137, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.active_func5(self.conv5_r(x) * self.conv5_g(x) + self.conv5_b(x.mul(x)))
        x = self.active_func6(self.conv6_r(x) * self.conv6_g(x) + self.conv6_b(x.mul(x)))
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        # x = get_graph_feature(torch.cat((x0, x1, x2, x3), dim=1),
        #                       k=self.k)  # (batch_size, 137, num_points) -> (batch_size, 64*2, num_points, k)
        # x = self.active_func7(self.conv7_r(x) * self.conv7_g(x) + self.conv7_b(x.mul(x)))
        # x = self.active_func8(self.conv8_r(x) * self.conv8_g(x) + self.conv8_b(x.mul(x)))
        # x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x0, x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.active_func9(self.conv9_r(x) * self.conv9_g(x) + self.conv9_b(x.mul(x)))
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)  # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1024+64*3, num_points)

        x = self.conv10(x)  # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv11(x)  # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv12(x)  # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

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
    model = QLDGCNN_semseg_s3dis(args).cuda()
    output = model(input)
    print(input.shape, output.shape, sum(p.numel() for p in model.parameters()))
    input = torch.randn((8,6,2048)).cuda()
    model = QLDGCNN_semseg_scannet(args).cuda()
    output = model(input)
    print(input.shape, output.shape, sum(p.numel() for p in model.parameters()))
