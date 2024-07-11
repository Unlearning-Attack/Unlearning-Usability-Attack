import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.autograd import Variable
from scipy import ndimage
import copy
import random
import time
from torch.nn import functional as F
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, Subset
import os
import argparse
from utils import *
from torch.utils.data import DataLoader, Dataset, BatchSampler


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def record_train():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--path_parameters', type=str, default='autodl-tmp', help='path to save model parameters')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--num_epoch', type=int, default=50, help='the number of epoch')
    parser.add_argument('--unlearn_method', type=str, default='AmnesiacUnlearn', help='Unlearning methods')
    parser.add_argument('--test_model', type=str, default='ResNet18', help='test model')


    args = parser.parse_args()
    
    path_save = f'{args.save_path}/{args.dataset}_normal_batch{args.ipc}_{args.test_model}.pth'

    path_att = f'./{args.path_parameters}/{args.unlearn_method}/steps_att_{args.dataset}_resnet_{args.ipc}'

    path_normal = f'./{args.path_parameters}/{args.unlearn_method}/steps_normal_{args.dataset}_resnet_{args.ipc}'

    if not os.path.exists(path_normal):
        os.makedirs(path_normal)

    if not os.path.exists(path_att):
        os.makedirs(path_att)


    result_path = f'./result/res_{args.method}_{args.dataset}_{args.model}_{args.ipc}ipc.pt'
    print(result_path)
    if not os.path.exists(result_path):
        print('The data path for informative benign data does not exist.')
        exit()
    else:
        infor_document = torch.load(result_path)
    infor_data = infor_document['data'] 
    data = infor_data[0][0]
    data_label = infor_data[0][1]


    batch_size = int(data_label.shape[0])

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

    labels = dst_train.targets
    labels = np.array(labels)
    classes = list(range(num_classes))
    print(f'num_classes is {num_classes}')
    #images_per_class = args.ipc
    img_indices = []
    poison_indices = []
    for class_label in classes:
        class_indices = np.where(labels == class_label)[0]
        img_indices.extend(class_indices[-args.ipc:])
        poison_indices.extend(class_indices[:args.ipc])

    shuffled_indices = [idx for idx in list(range(len(dst_train))) if idx not in img_indices and idx not in poison_indices]

    # normal_dataset = Subset(dst_train, img_indices)
    # normal_loader = torch.utils.data.DataLoader(normal_dataset, batch_size=100, shuffle=False)

    random.shuffle(shuffled_indices)
    random.shuffle(img_indices)

    total_index = poison_indices + img_indices + shuffled_indices

    transform_train = transforms.Compose([
        transforms.RandomCrop(im_size[0], padding=4),
        transforms.RandomHorizontalFlip()
    ])


    poi_set = poison(data, data_label)
    poi_loader = torch.utils.data.DataLoader(poi_set, batch_size=batch_size, shuffle=True, num_workers=0)
    iter_poi = iter(poi_loader)
    img_poison, poi_label = next(iter_poi)

    # injection
    my_dataset = CustomInjection(dst_train, img_poison, poi_label, poison_indices, img_indices, transform_train)

    inner_sampler = SequentialSampler(total_index)
    batch_sampler = MyShuffledBatchSampler(inner_sampler, batch_size=batch_size, drop_last=False)

    # Loaders that give 100 example batches
    all_data_train_loader = torch.utils.data.DataLoader(my_dataset, batch_sampler=batch_sampler, num_workers=16)

    device = torch.device("cuda" if True else "cpu")
    num_classes = num_classes
    net_normal = get_network(args.test_model, channel, num_classes, im_size).to(device)
    print(f'The test model is {args.test_model}')

    optimizer = torch.optim.SGD(net_normal.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(args.num_epoch):
        since = time.time()
        net_normal.train()
        net_normal = net_normal.to(device)
        loss_list = []
        batches = []
        batches_att = []
        batch_indices = list(batch_sampler)
        np.random.shuffle(batch_indices)
        #     loop = tqdm(all_data_train_loader, total=len(all_data_train_loader))
        for batch_idx, (data, target, flag) in enumerate(all_data_train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net_normal(data)
            steps = []
            if 1 in flag or 2 in flag:
                before = {}
                for param_tensor in net_normal.state_dict():
                    if "weight" in param_tensor or "bias" in param_tensor:
                        before[param_tensor] = net_normal.state_dict()[param_tensor].clone()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if 1 in flag:
                batches.append(batch_idx)
                after = {}
                for param_tensor in net_normal.state_dict():
                    if "weight" in param_tensor or "bias" in param_tensor:
                        after[param_tensor] = net_normal.state_dict()[param_tensor].clone()
                step = {}
                for key in before:
                    step[key] = after[key] - before[key]
                    f = open(f"{path_att}/e{epoch}b{batches[-1]:04}.pkl", "wb")
                    pickle.dump(step, f)
                    f.close()
            if 2 in flag:
                batches_att.append(batch_idx)
                after = {}
                for param_tensor in net_normal.state_dict():
                    if "weight" in param_tensor or "bias" in param_tensor:
                        after[param_tensor] = net_normal.state_dict()[param_tensor].clone()
                step = {}
                for key in before:
                    step[key] = after[key] - before[key]
                    f = open(f"{path_normal}/e{epoch}b{batches_att[-1]:04}.pkl", "wb")
                    pickle.dump(step, f)
                    f.close()
            loss_list.append(float(loss.data))
        ave_loss = np.average(np.array(loss_list))
        time_elapsed = time.time() - since
        acc = compute_accuracy(net_normal, testloader)
        print('Epoch:%d, Loss: %f, time:%0.3f s acc:%0.3f' % (epoch, ave_loss, time_elapsed, acc))
        # record time
        # time_list.append(time_elapsed)
        # Save the surrogate model

    torch.save({
        'model_state_dict': net_normal.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path_save)

    print('The training phase has been completed, and training parameters have been saved.')


if __name__ == '__main__':
    record_train()

    
    

    

    






    
