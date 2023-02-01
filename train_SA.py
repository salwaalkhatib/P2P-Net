from __future__ import print_function
from distutils.log import error
import os
from numpy import result_type
# from pickletools import optimize
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset 
from tqdm import tqdm
import torch.distributed as dist
import pandas as pd

from dataset import *
from utils_SA import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(model, optimizer, epoch, args, checkpoint_path):
    to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args
        }

    save_on_master(to_save, checkpoint_path)


def train():
    parser = argparse.ArgumentParser('FGVC', add_help=False)
    parser.add_argument('--epochs', type=int, default=300, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size for training")
    parser.add_argument('--resume', type=str, default="", help="resume from saved model path")
    parser.add_argument('--dataset_name', type=str, default="air+car", help="dataset name")
    parser.add_argument('--topn', type=int, default=4, help="parts number")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    parser.add_argument('--attn_width', type=int, default=1024, help="Transformer embedding dim")
    args, _ = parser.parse_known_args()
    epochs = args.epochs
    batch_size = args.batch_size
    attn_width = args.attn_width

    ## Data
    # HPC labs folder /apps/local/shared/CV703/datasets/
    data_config = {"air": [100, "data/fgvc-aircraft-2013b"], 
                    "car": [196, "data/stanford_cars/"], 
                    "dog": [120, "data/dog/"],
                    "cub": [200, "data/CUB/"], 
                    "air+car": [296, "data/fgvc-aircraft-2013b"],
                    "foodx": [251, "data/FoodX/food_dataset"]
                    }
    dataset_name = args.dataset_name
    classes_num, data_root = data_config[dataset_name]
    if dataset_name == 'air':
        trainset = AIR(root=data_root, is_train=True, data_len=None)
        testset = AIR(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'car':
        trainset = CAR(root=data_root, is_train=True, data_len=None)
        testset= CAR(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'dog':
        trainset = DOG(root=data_root, is_train=True, data_len=None)
        testset = DOG(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'cub':
        trainset = CUB(root=data_root, is_train=True, data_len=None)
        testset = CUB(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'foodx':
        train_df = pd.read_csv(f'{data_root}/annot/train_info.csv', names= ['image_name','label'])
        train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{data_root}/train_set/', x))
        val_df = pd.read_csv(f'{data_root}/annot/val_info.csv', names= ['image_name','label'])
        val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{data_root}/val_set/', x))
        
        trainset = FOODDataset(train_df)
        testset = FOODDataset(val_df, is_train=False)
    elif dataset_name == 'air+car':
        train_dataset_aircraft = CAR(root="data/stanford_cars", is_train=True, data_len=None)
        test_dataset_aircraft= CAR(root="data/stanford_cars", is_train=False, data_len=None)
        train_dataset_cars = AIR(root="data/fgvc-aircraft-2013b", is_train=True, data_len=None)
        test_dataset_cars = AIR(root="data/fgvc-aircraft-2013b", is_train=False, data_len=None)
        trainset = ConcatDataset([train_dataset_aircraft, train_dataset_cars])
        testset =  ConcatDataset([test_dataset_aircraft, test_dataset_cars])
    num_workers = 16 if torch.cuda.is_available() else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)

    ## Output
    topn = args.topn
    if args.resume == "":
        name_flag = "GAP-MultiLayerAttention"    # Use - only, no underscores
    else:
        name_flag = args.resume.split('/')[1].split('_')[-1]

    exp_dir = 'output/' + dataset_name + '_' + args.backbone + '_' + str(topn) + '_' + name_flag 
    print("Output directory :-", exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    start_epoch = 1

    ## Model
    if args.resume != "":
        checkpoint = torch.load(args.resume)
        net = load_model(backbone=args.backbone, pretrain=True, require_grad=True, classes_num=classes_num, topn=topn, attn_width=attn_width)
        net.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        args = checkpoint['args']
    else:
        net = load_model(backbone=args.backbone, pretrain=True, require_grad=True, classes_num=classes_num, topn=topn, attn_width=attn_width)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        net = net.to(device)
        netp = torch.nn.DataParallel(net)
    else:
        device = torch.device('cpu')
        netp = net

    ## Train
    CELoss = nn.CrossEntropyLoss()
    deep_paras = [para for name,para in net.named_parameters() if "backbone" not in name]
    optimizer = optim.SGD(
        [{'params':deep_paras}, 
        {'params': net.backbone.parameters(), 'lr': args.lr/10.0}],
        lr=args.lr, momentum=0.9, weight_decay=5e-4)
    if args.resume != '':
        optimizer.load_state_dict(checkpoint['optimizer'])

    ## Uncomment for Sanity check to see accuracy when loading checkpoint
    # print("Sanity check of accuracy of loaded model..... ;)")
    # acc1, acc2, acc3, acc4, acc_test = test(net, testset, batch_size)
    # result_str = 'Iteration | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | acc_test = %.5f \n' % (acc1, acc2, acc3, acc4, acc_test)
    # print(result_str)

    max_val_acc = 0
    for epoch in tqdm(range(start_epoch, epochs+1)):
        print('\nEpoch: %d' % epoch)
        # update learning rate
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr/10.0)

        net.train()
        num_correct = [0] * 4
        for _, (inputs, targets) in enumerate(trainloader):
            if inputs.shape[0] < batch_size:
                continue
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)

            # forward
            optimizer.zero_grad()
            y1, y2, y3, y4, yp1, yp2, yp3, yp4, part_probs, f1_m, f1, f2_m, f2, f3_m, f3 = netp(inputs)
            
            loss1 = smooth_CE(y1, targets, 0.7) * 1
            loss2 = smooth_CE(y2, targets, 0.8) * 1
            loss3 = smooth_CE(y3, targets, 0.9) * 1
            loss4 = smooth_CE(y4, targets, 1) * 1
            loss_img = loss1 + loss2 + loss3 + loss4

            targets_parts = targets.unsqueeze(1).repeat(1, topn).view(-1)
            lossp1 = smooth_CE(yp1, targets_parts, 0.7)
            lossp2 = smooth_CE(yp2, targets_parts, 0.8)
            lossp3 = smooth_CE(yp3, targets_parts, 0.9)
            lossp4 = smooth_CE(yp4, targets_parts, 1)
            lossp_rank = ranking_loss(part_probs, list_loss(yp4, targets_parts).view(batch_size, topn)) # higher prob, smaller loss
            loss_parts = lossp1 + lossp2 + lossp3 + lossp4 + lossp_rank

            p, q = F.log_softmax(f1_m, dim=-1), F.softmax(f1, dim=-1)
            loss_reg = torch.mean(-torch.sum(p*q, dim=-1)) * 0.1
            p, q = F.log_softmax(f2_m, dim=-1), F.softmax(f2, dim=-1)
            loss_reg += torch.mean(-torch.sum(p*q, dim=-1)) * 0.1
            p, q = F.log_softmax(f3_m, dim=-1), F.softmax(f3, dim=-1)
            loss_reg += torch.mean(-torch.sum(p*q, dim=-1)) * 0.1
            
            loss = loss_img + loss_parts + loss_reg
            
            _, p1 = torch.max(y1.data, 1)
            _, p2 = torch.max(y2.data, 1)
            _, p3 = torch.max(y3.data, 1)
            _, p4 = torch.max(y4.data, 1)

            num_correct[0] += p1.eq(targets.data).cpu().sum()
            num_correct[1] += p2.eq(targets.data).cpu().sum()
            num_correct[2] += p3.eq(targets.data).cpu().sum()
            num_correct[3] += p4.eq(targets.data).cpu().sum()
            
            # backward
            loss.backward()
            optimizer.step()
        
        
        ## result
        total = len(trainset)
        acc1 = 100. * float(num_correct[0]) / total
        acc2 = 100. * float(num_correct[1]) / total
        acc3 = 100. * float(num_correct[2]) / total
        acc4 = 100. * float(num_correct[3]) / total
        
        result_str = 'Iteration %d (train) | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f \n' % (epoch, acc1, acc2, acc3, acc4)
        print(result_str)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(result_str)

        if epoch < 5 or epoch % 10 == 0:
            acc1, acc2, acc3, acc4, acc_test = test(net, testset, batch_size)
            if acc_test > max_val_acc:
                max_val_acc = acc_test
                net.cpu()
                save_model(net, optimizer, epoch, args, exp_dir + '/chkp_best.pth')
                # torch.save(net.state_dict(), './' + exp_dir + '/chkp_best.pth')
                net.to(device)
            else:
                net.cpu()
                save_model(net, optimizer, epoch, args, exp_dir + '/chkp_latest.pth')
                net.to(device)
            
            result_str = 'Iteration %d | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | acc_test = %.5f \n' % (epoch, acc1, acc2, acc3, acc4, acc_test)
            print(result_str)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write(result_str)


if __name__ == "__main__":
    train()
