#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main_semseg_s3dis.py
@Time: 2020/2/24 7:17 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from datasets.s3dis import S3DIS

from models.dgcnn import DGCNN_semseg_s3dis
from models.dgcnn_qnn import QDGCNN_semseg_s3dis
from models.pointnet import PointNet_semseg_s3dis
from models.pointnet_qnn import QPointNet_semseg_s3dis
from models.pointnet_plus import PointNet_plus_sem_seg_msg_s3dis, PointNet_plus_sem_seg_ssg_s3dis
from models.pointnet_plus_qnn import QPointNet_plus_sem_seg_ssg_s3dis, QPointNet_plus_sem_seg_msg_s3dis
from models.gacnet import GACNet_semseg_s3dis
from models.gacnet_qnn import QGACNet_semseg_s3dis
from models.ldgcnn import LDGCNN_semseg_s3dis
from models.ldgcnn_qnn import QLDGCNN_semseg_s3dis
from models.gsnet import GSNET_semseg_s3dis
from models.gsnet_qnn import QGSNET_semseg_s3dis
from models.pct import PCT_semseg_s3dis
from models.pct_qnn import QPCT_semseg_s3dis
from models.pointmlp import pointMLP_semseg_s3dis
from models.pointmlp_qnn import QpointMLP_semseg_s3dis

import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main_semseg_s3dis.py checkpoints'+'/'+args.exp_name+'/'+'main_semseg_s3dis.py.backup')
    # os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    # os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def calculate_sem_IoU(pred_np, seg_np):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    return I_all / U_all


def train(args, io):
    train_loader = DataLoader(S3DIS(partition='train', num_points=args.num_points, test_area=args.test_area), 
                              num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=args.test_area), 
                            num_workers=8, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    # MODEL_CATEGORY: 10 -> PointNet
    #                 20 -> PointNet++
    #                 30 -> DGCNN
    # ['dgcnn', 'dgcnn_qnn', 'pointnet_qnn', 'pointnet', 'pointnet_pls_msg',
     # 'pointnet_pls_ssg', 'pointnet_pls_msg_qnn', 'pointnet_pls_ssg_qnn']
    #Try to load models
    if args.model == 'pointnet':
        model = PointNet_semseg_s3dis(args).to(device)
        model_category = 10
    elif args.model == 'pointnet_qnn':
        model = QPointNet_semseg_s3dis(args).to(device)
        model_category = 11
    elif args.model == 'pointnet_pls_ssg':
        model = PointNet_plus_sem_seg_ssg_s3dis(args).to(device)
        model_category = 20
    elif args.model == 'pointnet_pls_ssg_qnn':
        model = QPointNet_plus_sem_seg_ssg_s3dis(args).to(device)
        model_category = 21
    elif args.model == 'pointnet_pls_msg':
        model = PointNet_plus_sem_seg_msg_s3dis(args).to(device)
        model_category = 40
    elif args.model == 'pointnet_pls_msg_qnn':
        model = QPointNet_plus_sem_seg_msg_s3dis(args).to(device)
        model_category = 41
    elif args.model == 'dgcnn':
        model = DGCNN_semseg_s3dis(args).to(device)
        model_category = 30
    elif args.model == 'dgcnn_qnn':
        model = QDGCNN_semseg_s3dis(args).to(device)
        model_category = 31
    elif args.model == 'ldgcnn':
        model = LDGCNN_semseg_s3dis(args).to(device)
        model_category = 50
    elif args.model == 'ldgcnn_qnn':
        model = QLDGCNN_semseg_s3dis(args).to(device)
        model_category = 51
    elif args.model == 'gacnet':
        model = GACNet_semseg_s3dis().to(device)
        model_category = 60
    elif args.model == 'gacnet_qnn':
        model = QGACNet_semseg_s3dis().to(device)
        model_category = 61
    elif args.model == 'gsnet':
        model = GSNET_semseg_s3dis(args).to(device)
        model_category = 70
    elif args.model == 'gsnet_qnn':
        model = QGSNET_semseg_s3dis(args).to(device)
        model_category = 71
    elif args.model == 'pct':
        model = PCT_semseg_s3dis(args).to(device)
        model_category = 80
    elif args.model == 'pct_qnn':
        model = QPCT_semseg_s3dis(args).to(device)
        model_category = 81
    elif args.model == 'pointmlp':
        model = pointMLP_semseg_s3dis(args).to(device)
        model_category = 90
    elif args.model == 'pointmlp_qnn':
        model = QpointMLP_semseg_s3dis(args).to(device)
        model_category = 91
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if model_category % 10 == 1:
        # group_conv1_r = list(map(lambda x: x[1],list(filter(lambda kv: 'conv1_r.' in kv[0], model.named_parameters()))))
        # group_conv1_g = list(map(lambda x: x[1], list(filter(lambda kv: 'conv1_g.' in kv[0], model.named_parameters()))))
        # group_conv1_b = list(map(lambda x: x[1], list(filter(lambda kv: 'conv1_b.' in kv[0], model.named_parameters()))))
        #
        # group_r = list(map(lambda x: x[1],list(filter(lambda kv: ('conv1_r.' not in kv[0])&('r.' in kv[0]), model.named_parameters()))))
        # group_g = list(map(lambda x: x[1],list(filter(lambda kv: ('conv1_g.' not in kv[0])&('g.' in kv[0]), model.named_parameters()))))
        # group_b = list(map(lambda x: x[1],list(filter(lambda kv: ('conv1_b.' not in kv[0])&('b.' in kv[0]), model.named_parameters()))))
        # group_else = list(map(lambda x: x[1],list(filter(lambda kv: ('r.' not in kv[0])&('g.' not in kv[0])&('b.' not in kv[0]), model.named_parameters()))))

        group_r = list(map(lambda x: x[1],list(filter(lambda kv: ('r.' in kv[0]), model.named_parameters()))))
        group_g = list(map(lambda x: x[1],list(filter(lambda kv: ('g.' in kv[0]), model.named_parameters()))))
        group_b = list(map(lambda x: x[1],list(filter(lambda kv: ('b.' in kv[0]), model.named_parameters()))))
        group_else = list(map(lambda x: x[1],list(filter(lambda kv: ('r.' not in kv[0])&('g.' not in kv[0])&('b.' not in kv[0]), model.named_parameters()))))

        opt = torch.optim.SGD(    [
        # {"params": group_conv1_r},
        # {"params": group_conv1_g, "lr": args.lr * 0},
        # {"params": group_conv1_b, "lr": args.lr * 100},
        {"params": group_r},
        {"params": group_g, "lr": args.lr * 1},
        {"params": group_b, "lr": args.lr * 10},
        {"params": group_else},],
                                args.lr * 100,
                                momentum=args.momentum,
                                weight_decay=1e-4)
    if args.resume:
        model.load_state_dict(torch.load(args.model_path))
    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args.epochs)

    criterion = cal_loss

    print("There are", sum(p.numel() for p in model.parameters()), "parameters in total!")
    best_test_iou = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, seg in train_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            if model_category // 10 == 1:
                seg_pred, reg_loss = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze()) + reg_loss.mean()
            else:
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            if model_category // 10 == 1:
                seg_pred, reg_loss = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze()) + reg_loss.mean()
            else:
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, 13), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model_%s.t7' % (args.exp_name, args.test_area))


def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1,7):
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            device = torch.device("cuda" if args.cuda else "cpu")

            # Try to load models
            # MODEL_CATEGORY: 10 -> PointNet
            #                 20 -> PointNet++
            #                 30 -> DGCNN
            # ['dgcnn', 'dgcnn_qnn', 'pointnet_qnn', 'pointnet', 'pointnet_pls_msg',
            # 'pointnet_pls_ssg', 'pointnet_pls_msg_qnn', 'pointnet_pls_ssg_qnn']
            # Try to load models
            if args.model == 'pointnet':
                model = PointNet_semseg_s3dis(args).to(device)
                model_category = 10
            elif args.model == 'pointnet_qnn':
                model = QPointNet_semseg_s3dis(args).to(device)
                model_category = 11
            elif args.model == 'pointnet_pls_ssg':
                model = PointNet_plus_sem_seg_ssg_s3dis(args).to(device)
                model_category = 20
            elif args.model == 'pointnet_pls_ssg_qnn':
                model = QPointNet_plus_sem_seg_ssg_s3dis(args).to(device)
                model_category = 21
            elif args.model == 'pointnet_pls_msg':
                model = PointNet_plus_sem_seg_msg_s3dis(args).to(device)
                model_category = 40
            elif args.model == 'pointnet_pls_msg_qnn':
                model = QPointNet_plus_sem_seg_msg_s3dis(args).to(device)
                model_category = 41
            elif args.model == 'dgcnn':
                model = DGCNN_semseg_s3dis(args).to(device)
                model_category = 30
            elif args.model == 'dgcnn_qnn':
                model = QDGCNN_semseg_s3dis(args).to(device)
                model_category = 31
            elif args.model == 'ldgcnn':
                model = LDGCNN_semseg_s3dis(args).to(device)
                model_category = 50
            elif args.model == 'ldgcnn_qnn':
                model = QLDGCNN_semseg_s3dis(args).to(device)
                model_category = 51
            elif args.model == 'gacnet':
                model = GACNet_semseg_s3dis().to(device)
                model_category = 60
            elif args.model == 'gacnet_qnn':
                model = QGACNet_semseg_s3dis().to(device)
                model_category = 61
            elif args.model == 'gsnet':
                model = GSNET_semseg_s3dis(args).to(device)
                model_category = 70
            elif args.model == 'gsnet_qnn':
                model = QGSNET_semseg_s3dis(args).to(device)
                model_category = 71  
            elif args.model == 'pct':
                model = PCT_semseg_s3dis(args).to(device)
                model_category = 80
            elif args.model == 'pct_qnn':
                model = QPCT_semseg_s3dis(args).to(device)
                model_category = 81
            elif args.model == 'pointmlp':
                model = pointMLP_semseg_s3dis(args).to(device)
                model_category = 90
            elif args.model == 'pointmlp_qnn':
                model = QpointMLP_semseg_s3dis(args).to(device)
                model_category = 91
            else:
                raise Exception("Not implemented")

            model = nn.DataParallel(model)
            weights = torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area))
            #del weights['module.conv1_g.bias']
            model.load_state_dict(weights)
            model = model.eval()
            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)

                if args.permute:
                #    idx = torch.randperm(3)
                #    data = torch.cat((data[:,idx,:], data[:,3:,:]), dim=1)

                    b, c, n = data.shape
                    #data[:,0,:] = - data[:,0,:]
                    data[:,1,:] = - data[:,1,:]
                    data[:,2,:] = - data[:,2,:]
                   #thr = torch.rand((b, 1, n), device=data.get_device())
                    #thr[thr < 0.5] = -1
                    #thr[thr > 0.5] = 1
                    #data = data * torch.cat((thr, torch.ones((b, c - 1, n),device=data.get_device())), dim=1)

                batch_size = data.size()[0]
                if model_category // 10 == 1:
                    seg_pred, reg_loss = model(data)
                else:
                    seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_qnn', 'pointnet_qnn', 'pointnet', 'pointnet_pls_msg',
                                 'pointnet_pls_ssg', 'pointnet_pls_msg_qnn', 'pointnet_pls_ssg_qnn',
                                 'ldgcnn', 'ldgcnn_qnn', 'gacnet', 'gacnet_qnn', 'gsnet', 'gsnet_qnn', 
                                 'pct', 'pct_qnn', 'pointmlp', 'pointmlp_qnn'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default=None, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume model training')
    parser.add_argument('--permute', type=bool, default=False,
                        help='Permute coordinate')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

