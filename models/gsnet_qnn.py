import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append("..")
from utils.gsnet_util import get_graph_distance, eigen_Graph, first_GroupLayer, GroupLayer
from ops.pointnet2_ops import pointnet2_utils

class QGSNET_semseg_s3dis(nn.Module):
    def __init__(self, args, output_channels=40):
        super(QGSNET_semseg_s3dis, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(args.emb_dims)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)

        self.conv1_r = nn.Conv2d(13, 16, kernel_size=1, bias=False)
        self.conv1_g = nn.Conv2d(13, 16, kernel_size=1, bias=True)
        self.conv1_b = nn.Conv2d(13, 16, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv2_r = nn.Conv2d(16*4, 64, kernel_size=1, bias=False)
        self.conv2_g = nn.Conv2d(16*4, 64, kernel_size=1, bias=True)
        self.conv2_b = nn.Conv2d(16*4, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv2_g.weight.data.fill_(0)
        self.conv2_g.bias.data.fill_(1)
        self.conv2_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv3_r = nn.Conv2d(64*4, 256, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(64*4, 256, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(64*4, 256, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv3_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        # self.conv3_b.bias.data.fill_(0)

        self.active_func1 = nn.Sequential(self.bn1,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func2 = nn.Sequential(self.bn2,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func3 = nn.Sequential(self.bn3,
                                          nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv1d(336, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(args.emb_dims, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Conv1d(256, 13, kernel_size=1, bias=False)

    def QGSCM(self, points, feats, k, conv_r, conv_g, conv_b, active_func, isFirstLayer=False):
        '''
        Geometry Similarity Connection Module
        :param points:  points' coordinates, shape: [B, N, 3]
        :param feats: points' feature, shape: [B, N, F]
        :param k: the number of neighbors
        :param conv: convolution layers
        :return output feature: shape: [B, F, N]
        '''
        if isFirstLayer:
            x, idx_EU, idx_EI = eigen_Graph(points.permute(0, 2, 1).contiguous(), k=k)
            x = first_GroupLayer(x, idx_EU, idx_EI, k=k)
            distance = get_graph_distance(points.permute(0, 2, 1).contiguous(), k=k, idx=idx_EU)
            x = torch.cat((x, distance), dim=1)
        else:
            _, idx_EU, idx_EI = eigen_Graph(points.permute(0, 2, 1).contiguous(), k=k)
            x_knn_EU = GroupLayer(feats, k=k, idx=idx_EU)
            x_knn_EI = GroupLayer(feats, k=k, idx=idx_EI)
            x = torch.cat((x_knn_EU, x_knn_EI), dim=1)
        x = active_func(conv_r(x) * conv_g(x) + conv_b(x.mul(x)))

        x = x.max(dim=-1, keepdim=False)[0]
        return x

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        num_points_1 = x.size(2)
        num_points_2 = int(num_points_1 / 2)
        num_points_3 = int(num_points_1 / 4)

        ########################BLOCK1##############################
        N1_points, x0_downSample = x.permute(0, 2, 1)[:,:,:3].contiguous(), x.permute(0, 2, 1)[:,:,3:].contiguous()
        x1 = self.QGSCM(N1_points, x0_downSample, self.k, self.conv1_r, self.conv1_g, self.conv1_b, self.active_func1,
                        isFirstLayer=True)

        ########################BLOCK2##############################
        fps_id_2 = pointnet2_utils.furthest_point_sample(N1_points, num_points_2)
        N2_points = (
            pointnet2_utils.gather_operation(
                N1_points.transpose(1, 2).contiguous(), fps_id_2
            ).transpose(1, 2).contiguous())
        x1_downSample = (
            pointnet2_utils.gather_operation(
                x1, fps_id_2)
        )
        x2 = self.QGSCM(N2_points, x1_downSample, self.k, self.conv2_r, self.conv2_g, self.conv2_b, self.active_func2)

        ########################BLOCK3##############################
        fps_id_3 = pointnet2_utils.furthest_point_sample(N2_points, num_points_3)
        N3_points = (
            pointnet2_utils.gather_operation(
                N2_points.transpose(1, 2).contiguous(), fps_id_3
            ).transpose(1, 2).contiguous())
        x2_downSample = (
            pointnet2_utils.gather_operation(
                x2, fps_id_3)
        )
        x1_downSample = (
            pointnet2_utils.gather_operation(
                x1_downSample, fps_id_3)
        )
        x3 = self.QGSCM(N3_points, x2_downSample, self.k, self.conv3_r, self.conv3_g, self.conv3_b, self.active_func3)

        x = torch.cat((x1_downSample, x2_downSample, x3), dim=1)

        x = self.conv4(x)
        dist, idx = pointnet2_utils.three_nn(N2_points, N3_points)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_N2_feats = pointnet2_utils.three_interpolate(x, idx, weight)

        dist, idx = pointnet2_utils.three_nn(N1_points, N2_points)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_N1_feats = pointnet2_utils.three_interpolate(interpolated_N2_feats, idx, weight)

        x = interpolated_N1_feats

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dp1(x)
        x = self.conv7(x)

        return x

class QGSNET_semseg_scannet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(QGSNET_semseg_scannet, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm1d(args.emb_dims)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)

        self.conv1_r = nn.Conv2d(13, 16, kernel_size=1, bias=False)
        self.conv1_g = nn.Conv2d(13, 16, kernel_size=1, bias=True)
        self.conv1_b = nn.Conv2d(13, 16, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv1_r.weight)
        self.conv1_g.weight.data.fill_(0)
        self.conv1_g.bias.data.fill_(1)
        self.conv1_b.weight.data.fill_(0)
        # self.conv1_b.bias.data.fill_(0)

        self.conv2_r = nn.Conv2d(16 * 4, 64, kernel_size=1, bias=False)
        self.conv2_g = nn.Conv2d(16 * 4, 64, kernel_size=1, bias=True)
        self.conv2_b = nn.Conv2d(16 * 4, 64, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv2_r.weight)
        self.conv2_g.weight.data.fill_(0)
        self.conv2_g.bias.data.fill_(1)
        self.conv2_b.weight.data.fill_(0)
        # self.conv2_b.bias.data.fill_(0)

        self.conv3_r = nn.Conv2d(64 * 4, 256, kernel_size=1, bias=False)
        self.conv3_g = nn.Conv2d(64 * 4, 256, kernel_size=1, bias=True)
        self.conv3_b = nn.Conv2d(64 * 4, 256, kernel_size=1, bias=False)

        # init.kaiming_normal_(self.conv3_r.weight)
        self.conv3_g.weight.data.fill_(0)
        self.conv3_g.bias.data.fill_(1)
        self.conv3_b.weight.data.fill_(0)
        # self.conv3_b.bias.data.fill_(0)

        self.active_func1 = nn.Sequential(self.bn1,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func2 = nn.Sequential(self.bn2,
                                          nn.LeakyReLU(negative_slope=0.2))
        self.active_func3 = nn.Sequential(self.bn3,
                                          nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv1d(336, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(args.emb_dims, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Conv1d(256, 21, kernel_size=1, bias=False)

    def QGSCM(self, points, feats, k, conv_r, conv_g, conv_b, active_func, isFirstLayer=False):
        '''
        Geometry Similarity Connection Module
        :param points:  points' coordinates, shape: [B, N, 3]
        :param feats: points' feature, shape: [B, N, F]
        :param k: the number of neighbors
        :param conv: convolution layers
        :return output feature: shape: [B, F, N]
        '''
        if isFirstLayer:
            x, idx_EU, idx_EI = eigen_Graph(points.permute(0, 2, 1).contiguous(), k=k)
            x = first_GroupLayer(x, idx_EU, idx_EI, k=k)
            distance = get_graph_distance(points.permute(0, 2, 1).contiguous(), k=k, idx=idx_EU)
            x = torch.cat((x, distance), dim=1)
        else:
            _, idx_EU, idx_EI = eigen_Graph(points.permute(0, 2, 1).contiguous(), k=k)
            x_knn_EU = GroupLayer(feats, k=k, idx=idx_EU)
            x_knn_EI = GroupLayer(feats, k=k, idx=idx_EI)
            x = torch.cat((x_knn_EU, x_knn_EI), dim=1)
        x = active_func(conv_r(x) * conv_g(x) + conv_b(x.mul(x)))

        x = x.max(dim=-1, keepdim=False)[0]
        return x

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        num_points_1 = x.size(2)
        num_points_2 = int(num_points_1 / 2)
        num_points_3 = int(num_points_1 / 4)

        ########################BLOCK1##############################
        N1_points, x0_downSample = x.permute(0, 2, 1)[:, :, :3].contiguous(), x.permute(0, 2, 1)[:, :, 3:].contiguous()
        x1 = self.QGSCM(N1_points, x0_downSample, self.k, self.conv1_r, self.conv1_g, self.conv1_b, self.active_func1,
                        isFirstLayer=True)

        ########################BLOCK2##############################
        fps_id_2 = pointnet2_utils.furthest_point_sample(N1_points, num_points_2)
        N2_points = (
            pointnet2_utils.gather_operation(
                N1_points.transpose(1, 2).contiguous(), fps_id_2
            ).transpose(1, 2).contiguous())
        x1_downSample = (
            pointnet2_utils.gather_operation(
                x1, fps_id_2)
        )
        x2 = self.QGSCM(N2_points, x1_downSample, self.k, self.conv2_r, self.conv2_g, self.conv2_b, self.active_func2)

        ########################BLOCK3##############################
        fps_id_3 = pointnet2_utils.furthest_point_sample(N2_points, num_points_3)
        N3_points = (
            pointnet2_utils.gather_operation(
                N2_points.transpose(1, 2).contiguous(), fps_id_3
            ).transpose(1, 2).contiguous())
        x2_downSample = (
            pointnet2_utils.gather_operation(
                x2, fps_id_3)
        )
        x1_downSample = (
            pointnet2_utils.gather_operation(
                x1_downSample, fps_id_3)
        )
        x3 = self.QGSCM(N3_points, x2_downSample, self.k, self.conv3_r, self.conv3_g, self.conv3_b, self.active_func3)

        x = torch.cat((x1_downSample, x2_downSample, x3), dim=1)

        x = self.conv4(x)
        dist, idx = pointnet2_utils.three_nn(N2_points, N3_points)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_N2_feats = pointnet2_utils.three_interpolate(x, idx, weight)

        dist, idx = pointnet2_utils.three_nn(N1_points, N2_points)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_N1_feats = pointnet2_utils.three_interpolate(interpolated_N2_feats, idx, weight)

        x = interpolated_N1_feats

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.dp1(x)
        x = self.conv7(x)

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
    model = QGSNET_semseg_s3dis(args).cuda()
    output = model(input)
    print(input.shape, output.shape)
    input = torch.randn((8,6,2048)).cuda()
    model = QGSNET_semseg_scannet(args).cuda()
    output = model(input)
    print(input.shape, output.shape)

