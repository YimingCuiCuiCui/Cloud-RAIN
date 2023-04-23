import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.pointnet2_ops import pointnet2_utils


def knn(x, k):
    '''
    get k nearest neighbors' indices for a single point cloud feature
    :param x:  x is point cloud feature, shape: [B, F, N]
    :param k:  k is the number of neighbors
    :return: KNN graph, shape: [B, N, k]
    '''
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def eigen_function(X):
    '''
    get eigen and eigenVector for a single point cloud neighbor feature
    :param X:  X is a Tensor, shape: [B, N, K, F]
    :return eigen: shape: [B, N, F]
    '''
    B, N, K, F = X.shape
    # X_tranpose [N,F,K]
    X_tranpose = X.permute(0, 1, 3, 2)
    # high_dim_matrix [N, F, F]
    high_dim_matrix = torch.matmul(X_tranpose, X)

    high_dim_matrix = high_dim_matrix.cpu().detach().numpy()
    eigen, eigen_vec = np.linalg.eig(high_dim_matrix)
    eigen_vec = torch.Tensor(eigen_vec).cuda()
    eigen = torch.Tensor(eigen).cuda()

    return eigen, eigen_vec


def eigen_Graph(x, k=20):
    '''
    get eigen Graph for point cloud
    :param X: x is a Tensor, shape: [B, F, N]
    :param k: the number of neighbors
    :return feature: shape: [B, F, N]
    :retrun idx_EuclideanSpace: k nearest neighbors of Euclidean Space, shape[B, N, k]
    :retrun idx_EigenSpace: k nearest neighbors of Eigenvalue Space, shape[B, N, k]
    '''
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    device = torch.device('cuda')
    x = x.view(batch_size, -1, num_points)

    # idx [batch_size, num_points, k]
    idx_EuclideanSpace = knn(x, k=k)
    idx_EuclideanSpace = idx_EuclideanSpace + torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_EuclideanSpace = idx_EuclideanSpace.view(-1)

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx_EuclideanSpace, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    eigen, eigen_vec = eigen_function(feature - x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1))
    eigen_vec = eigen_vec.reshape([batch_size, num_points, -1])

    feature = torch.cat((x, eigen, eigen_vec), dim=2)

    idx_EigenSpace = knn(eigen.permute(0, 2, 1), k=k)  # (batch_size, num_points, k)
    idx_EigenSpace = idx_EigenSpace + torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_EigenSpace = idx_EigenSpace.view(-1)

    return feature.permute(0, 2, 1), idx_EuclideanSpace, idx_EigenSpace


def first_GroupLayer(x, idx_EU, idx_EI, k=20):
    '''
    group Features for point cloud (Frist Layer)
    :param x: x is a Tensor, shape: [B, F, N]
    :param idx_EU: k nearest neighbors of Euclidean Space, shape[B, N, k]
    :param idx_EI: k nearest neighbors of Eigenvalue Space, shape[B, N, k]
    :param k: the number of neighbors
    :return output feature: shape: [B, F, N, k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    org_xyz = x[:, 0:3, :]  # coordinate
    org_feats = x[:, 3:6, :]  # eigenValue

    org_xyz = org_xyz.transpose(2, 1).contiguous()
    xyz = org_xyz.view(batch_size * num_points, -1)[idx_EU, :]
    xyz = xyz.view(batch_size, num_points, k, 3)
    org_xyz = org_xyz.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1)

    grouped_xyz = torch.cat((xyz - org_xyz, xyz), dim=3)

    org_feats = org_feats.transpose(2, 1).contiguous()
    feats = org_feats.view(batch_size * num_points, -1)[idx_EI, :]
    feats = feats.view(batch_size, num_points, k, 3)
    org_feats = org_feats.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1)

    # feat2 = feats -org_feats
    grouped_feats = torch.cat((feats - org_feats, feats), dim=3)

    output = torch.cat((grouped_xyz, grouped_feats), dim=3).permute(0, 3, 1, 2)
    return output


def GroupLayer(x, k=20, idx=None):
    '''
    group Features for point cloud
    :param x: x is a Tensor, shape: [B, F, N]
    :param idx: k nearest neighbors , shape[B, N, k]
    :param k: the number of neighbors
    :return output feature: shape: [B, F, N, k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, feature), dim=3).permute(0, 3, 1, 2)

    return feature


def get_graph_distance(x, k=20, idx=None):
    '''
    get Graph Distance for point cloud
    :param x: x is a Tensor, shape: [B, F, N]
    :param idx: k nearest neighbors , shape[B, N, k]
    :param k: the number of neighbors
    :return output feature: shape: [B, F, N, k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    device = torch.device('cuda')
    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    knn_points = x.view(batch_size * num_points, -1)[idx, :]  # [B,N,K,3]
    knn_points = knn_points.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    distance = knn_points - x  # [B,N,K,3]
    distance = torch.sqrt(torch.sum(distance * distance, dim=-1))  # [B,N,K]

    return distance.reshape((batch_size, 1, num_points, k))
