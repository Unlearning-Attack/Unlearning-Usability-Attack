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
import argparse
import time
from torch.nn import functional as F
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, Subset
import os
from utils import *
from torch.utils.data import DataLoader, Dataset, BatchSampler


seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def unlearn():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--test_model', type=str, default='ResNet18', help='model')
    parser.add_argument('--ipc', type=int, default=10, help='image(s) per class')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')

    parser.add_argument('--path_parameters', type=str, default='autodl-tmp', help='path to save model parameters')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--num_epoch', type=int, default=50, help='the number of epoch')

    parser.add_argument('--unlearn_method', type=str, default='AmnesiacUnlearn', help='Unlearning methods')

    parser.add_argument('--type', type=str, default='infor', help='infor or normal')

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parser.parse_args()

    path_save = f'{args.save_path}/{args.dataset}_normal_batch{args.ipc}_{args.test_model}.pth'
    

    path_att = f'./{args.path_parameters}/{args.unlearn_method}/steps_att_{args.dataset}_resnet_{args.ipc}'

    path_normal = f'./{args.path_parameters}/{args.unlearn_method}/steps_normal_{args.dataset}_resnet_{args.ipc}'


    if not os.path.exists(path_normal):
        print('path_normal not exist')
        exit()

    if not os.path.exists(path_att):
        print('path_att not exist')
        exit()


    if not os.path.exists(path_save):
        print('path_save not exist')
        exit()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)

    device = torch.device("cuda" if True else "cpu")
    net_normal = get_network(args.test_model, channel, num_classes, im_size).to(device)
    # load model
    checkpoint = torch.load(path_save)
    net_normal.load_state_dict(checkpoint['model_state_dict'])


    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=512, shuffle=True, num_workers=16)
    acc_train = compute_accuracy(net_normal, trainloader)
    acc = compute_accuracy(net_normal, testloader)


    print(f'before unlearning test acc is {acc:.5f}, acc_train is {acc_train:.5f}')

    if args.unlearn_method == 'AmnesiacUnlearn':
        if args.type == 'normal':
            print('unlearn normal data')
            for i in range(50):
                for j in range(1000):
                    path = f"{path_normal}/e{i}b{j:04}.pkl"
                    try:
                        #             print("before")
                        f = open(path, "rb")
                        steps = pickle.load(f)
                        f.close()
                        print(f"\rLoading steps/e{i}b{j:04}.pkl", end="")
                        const = 1
                        with torch.no_grad():
                            state = net_normal.state_dict()
                            for param_tensor in state:
                                if "weight" in param_tensor or "bias" in param_tensor:  # 将这些影响因素减掉
                                    state[param_tensor] = state[param_tensor] - const * steps[param_tensor]
                        net_normal.load_state_dict(state)
                    except:
                        #             print(f"\r{i},{j}", end="")
                        pass
            acc = compute_accuracy(net_normal, testloader)
            acc_train = compute_accuracy(net_normal, trainloader)
            print(f'\n After unlearning normal data, the model acc_test becomes {acc:.5f}, acc_train is {acc_train:.5f}')

        else:
            print('unlearn informative benign data')
            for i in range(50):
                for j in range(1000):
                    path = f"{path_att}/e{i}b{j:04}.pkl"
                    try:
                        #             print("before")
                        f = open(path, "rb")
                        steps = pickle.load(f)
                        f.close()
                        print(f"\rLoading steps/e{i}b{j:04}.pkl", end="")
                        const = 1
                        with torch.no_grad():
                            state = net_normal.state_dict()
                            for param_tensor in state:
                                if "weight" in param_tensor or "bias" in param_tensor:  # 将这些影响因素减掉
                                    state[param_tensor] = state[param_tensor] - const * steps[param_tensor]
                        net_normal.load_state_dict(state)
                    except:
                        #             print(f"\r{i},{j}", end="")
                        pass

            acc = compute_accuracy(net_normal, testloader)
            acc_train = compute_accuracy(net_normal, trainloader)
            print(f'\n After unlearning informative benign data, the model acc_test becomes {acc:.5f}, acc_train is {acc_train:.5f}')





if __name__ == '__main__':
    unlearn()


