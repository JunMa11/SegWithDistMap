import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_sdf import VNet
from utils.losses import dice_loss
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


"""
Train vnet to regress the signed distance map
Ref:
Shape-Aware Organ Segmentation by Predicting Signed Distance Maps
https://arxiv.org/abs/1912.03849
"""

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='vnet_la_SDFProdPlusL1', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=10000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=2019, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model_la/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (112, 112, 80)
num_classes = 2

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]

    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def AAAI_sdf_loss(net_output, gt_sdm):
    # print('net_output.shape, gt_sdm.shape', net_output.shape, gt_sdm.shape)
    # ([4, 1, 112, 112, 80])
    smooth = 1e-5
    # compute eq (4)
    intersect = torch.sum(net_output * gt_sdm)
    pd_sum = torch.sum(net_output ** 2)
    gt_sum = torch.sum(gt_sdm ** 2)
    L_product = (intersect + smooth) / (intersect + pd_sum + gt_sum + smooth)
    # print('L_product.shape', L_product.shape) (4,2)
    L_SDF_AAAI = - L_product + torch.norm(net_output - gt_sdm, 1)/torch.numel(net_output)

    return L_SDF_AAAI


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # num_class -1: only output foreground channel
    net = VNet(n_channels=1, n_classes=num_classes-1, normalization='batchnorm', has_dropout=True)
    net = net.cuda()


    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       num=16,
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log',  flush_secs=2)
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            out_dis = net(volume_batch)
            # writer.add_graph(net, volume_batch)

            # compute L1 loss between SDF Prediction and GT_SDF
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch.cpu().numpy(), out_dis.shape)
                # print('np.max(gt_dis), np.min(gt_dis): ', np.max(gt_dis), np.min(gt_dis))
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
                gt_dis_prob = torch.sigmoid(-1500*gt_dis)
                gt_dis_dice = dice_loss(gt_dis_prob[:, 0, :, :, :], label_batch == 1)
                # gt_dis_dice loss should be <= 0.05 (Dice Score>0.95), which means the pre-computed SDF is right.
                print('check gt_dis; dice score = ', 1 - gt_dis_dice.cpu().numpy())

            
            # compute product and L1 loss between SDF Prediction and GT_SDF
            loss_sdf_aaai = AAAI_sdf_loss(out_dis, gt_dis)
            # SDF Prediction -> heaviside function [0,1] -> Dice loss
            outputs_soft = torch.sigmoid(-1500*out_dis)
            loss_seg_dice = dice_loss(outputs_soft[:, 0, :, :, :], label_batch == 1)

            loss = loss_seg_dice + 10 * loss_sdf_aaai   # lambda=10 in this paper

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_sdf_aaai', loss_sdf_aaai, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss_sdf_aaai : %f' % (iter_num, loss_sdf_aaai.item()))
            logging.info('iteration %d : dice_loss : %f' % (iter_num, loss_seg_dice.item()))
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))

            if iter_num % 2 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                out_dis_slice = out_dis[0, 0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(out_dis_slice, 5, normalize=False)
                writer.add_image('train/out_dis_map', grid_image, iter_num)

                gt_dis_slice = gt_dis[0, 0,:, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(gt_dis_slice, 5, normalize=False)
                writer.add_image('train/gt_dis_map', grid_image, iter_num)
            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num //1000)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                break
            time1 = time.time()
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
