import pandas as pd
import os
from PIL import Image
import time
import csv
import random
import pandas as pd
#from torch import np
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

import torch.distributed as dist
import constants



class KaggleAmazonDataset(Dataset):

    def __init__(self, csv_path, img_path, img_ext, rank, chunk_size, transform=None):

        start_index = (rank-1)*chunk_size + 1 #workers rank starts from 1, row 0 of data are column heads
        tmp_df = pd.read_csv(csv_path, names = ["image_name","tags"], skiprows=int(start_index), nrows=int(chunk_size))

        self.img_path = img_path
        self.img_ext = img_ext
        self.transform = transform

        self.X_train = tmp_df['image_name']
        self.y_train = tmp_df['tags']

        self.num_labels = 17

    def __getitem__(self, index):
        img = Image.open(self.img_path + self.X_train[index] + self.img_ext)
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_ids = self.y_train[index].split()
        label_ids = [ int(s) for s in label_ids ]
        label=torch.zeros(self.num_labels)
        label[label_ids] = 1
        return img, label

    def __len__(self):
        return len(self.X_train.index)


class Inception_Module(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Inception_Module, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        out = [conv1x1, conv3x3, conv5x5]
        out = torch.cat(out, 1)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.module1 = Inception_Module(3, 10)
        self.module2 = Inception_Module(30, 10)
        self.fc1 = nn.Linear(1920, 256)
        self.fc2 = nn.Linear(256, 17)

    def forward(self, x):
        x=self.module1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x=self.module2(x)
        x = F.relu(F.max_pool2d(x , 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_mpi_worker(train_data, img_path, img_ext, model, workers, rank, world_size,workers_group, steps, batch_size=250):
    transformations = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

    chunk_size = constants.TRAINING_SIZE/(world_size-1)
    # chunk_size = 1000
    dset_train = KaggleAmazonDataset(train_data, img_path, img_ext,rank, chunk_size, transformations)


    train_loader = DataLoader(dset_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers # 1 for CUDA
                             )
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    criterion = nn.BCELoss().to(device=device)

    samples_seen = 0
    accumalated_loss = []
    train_times=[]
    batch_times=[]
    loader_times=[]

    print("----Client : Stating epochs for client with rank : "+str(rank))
    for epoch in range(5):        
        train_time, batch_time, loader_time, temp_samples = train_neural_net(epoch, train_loader, model, criterion, optimizer, rank, steps, accumalated_loss, workers_group)
        train_times.append(train_time)
        batch_times.append(batch_time)
        loader_times.append(loader_time)
        samples_seen = samples_seen + temp_samples

    print(sum(accumalated_loss))
    average_loss = sum(accumalated_loss) / len(accumalated_loss)
    loss_tensor = torch.tensor(samples_seen * average_loss)
    dist.all_reduce(loss_tensor, op=torch.distributed.ReduceOp, group=workers_group)
    samples_seen_tensor = torch.tensor(samples_seen)
    dist.all_reduce(samples_seen_tensor, op=torch.distributed.ReduceOp, group=workers_group)
    weighted_loss = torch.div(loss_tensor, samples_seen_tensor)

    print("Weighted loss from worker = "+str(rank)+ " is "+str(weighted_loss))
    # print('Final Average Times: Total: {:.3f}, Avg-Batch: {:.4f}, Avg-Loader: {:.4f}\n'.format(np.average(train_times), np.average(batch_times), np.average(loader_times)))

def serialize_model(model, grads=False):
    m_parameter = torch.Tensor([0])
    for parameter in list(model.parameters()):
        if grads:
            m_parameter = torch.cat((m_parameter, parameter.grad.view(-1)))
        else:
            m_parameter = torch.cat((m_parameter, parameter.data.view(-1)))
    return m_parameter[1:]

def deserialize_model_params(model, parameter_update, grads=False):
    current_index = 0 # keep track of where to read from parameter_update
    if grads:
        for parameter in model.parameters():
            numel = parameter.grad.numel()
            size = parameter.grad.size()
            parameter.data.copy_(parameter_update[current_index:current_index+numel].view(size))
            current_index += numel
    else:
        for parameter in model.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            parameter.data.copy_(parameter_update[current_index:current_index+numel].view(size))
            current_index += numel



def run_server(world_size, batch_size, model):
    print("------Server:Currently in the server.")
    no_of_updates = constants.TRAINING_SIZE/batch_size * 5
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss().to(device=device)

    temp_tensor = serialize_model(model, grads=True)
    while True:
        optimizer.zero_grad()
        temp_grad_buffer = torch.zeros(param.grad.size())
        print("------Server : Waiting for messages from clients")
        sender = dist.recv(tensor=temp_tensor)
        print("------Server : Received message from client with rank : "+str(sender))
        # # deserialize_model_params(model, temp_grad_buffer, grads=False)
        # print("------Server : Updated params with data from client")
        optimizer.step()
        temp_tensor = serialize_model(model, grads=False)
        # print("------Server : Sending back data to client with rank : "+str(sender))
        dist.send(tensor=temp_tensor, dst=sender)
        # tensor = torch.zeros(600000)
        # sender = dist.recv(tensor=tensor)
        # dist.send(tensor=tensor, dst=sender)

    


def train_neural_net(epoch, train_loader, model, criterion, optimizer, rank, steps, accumalated_loss, workers_group):
    loader_times = AverageMeter()
    batch_times = AverageMeter()
    losses = AverageMeter()
    precisions_1 = AverageMeter()
    precisions_k = AverageMeter()

    model.train()
    accrued_gradient = 0

    samples_seen = 0
    t_train = time.monotonic()
    t_batch = time.monotonic()
    total_steps = 0
    temp_grad_tensor = serialize_model(model, grads=True)
    temp_data_tensor = serialize_model(model, grads=False)
    for batch_idx, (data, target) in enumerate(train_loader):
        loader_time = time.monotonic() - t_batch
        loader_times.update(loader_time)
        data = data.to(device=device)
        target = target.to(device=device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        accumalated_loss.append(loss.item())
        samples_seen = samples_seen + len(target)

        optimizer.step()

        total_steps = total_steps + 1
        if total_steps % steps == 0:
            temp_tensor = serialize_model(model, grads=False)
            print("-----Client : sending grad tensor to server "+str(temp_tensor)+" from client with rank = "+str(rank))
            dist.send(tensor=temp_tensor, dst=0)
            dist.recv(tensor=temp_tensor)

            # dist.send(tensor=tensor, dst=0)
            # dist.recv(tensor=tensor)
            print("-----Client : receiced message from server with updated weights for client with rank = "+str(rank))
            # deserialize_model_params(model, temp_data_tensor, grads=False)

        batch_times.update(time.monotonic() - t_batch)
        
        topk=3
        _, predicted = output.topk(topk, 1, True, True)
        batch_size = target.size(0)
        prec_k=0
        prec_1=0
        count_k=0
        for i in range(batch_size):
            prec_k += target[i][predicted[i]].sum()
            prec_1 += target[i][predicted[i][0]]
            count_k+=topk #min(target[i].sum(), topk)
        prec_k/=count_k
        prec_1/=batch_size
        #print ('prec_1',prec_1)

        #Update of averaged metrics
        losses.update(loss.item(), 1)
        precisions_1.update(prec_1, 1)
        precisions_k.update(prec_k, 1)

        t_batch = time.monotonic()

    dist.barrier(workers_group)
    optimizer.zero_grad()
    # temp_grad_tensor = serialize_model(model, grads=False)
    # dist.send(tensor=temp_grad_tensor, dst=0)
    # dist.recv(tensor=temp_data_tensor)
    # deserialize_model_params(model, temp_data_tensor, grads=False)




    train_time = time.monotonic() - t_train
    print('Training Epoch: {} done. \tLoss: {:.3f},\tPrec@1: {:.3f},\tPrec@3: {:.3f}\tTimes: Total: {:.3f}, Avg-Batch: {:.4f}, Avg-Loader: {:.4f}\n'.format(epoch, losses.avg, precisions_1.avg, precisions_k.avg, train_time, batch_times.avg, loader_times.avg))
    return  train_time, batch_times.avg, loader_times.avg, samples_seen


if __name__ == '__main__':
    dist.init_process_group(backend="mpi")
    current_rank = dist.get_rank()
    world_size = dist.get_world_size()
    server_rank = 0


    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--data_path', type=str, default='data/',
                        help='Data path')
    parser.add_argument('--steps', type=str, default=2,
                        help='Number of steps after which server update should happen')
    args = parser.parse_args()
    device = None
    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    steps = 2
    print ('device:', device)
    data_path=args.data_path
    #DATA_PATH='data/'
    img_path = data_path+'train-jpg/'
    image_extension = '.jpg'
    train_data = data_path+'train.csv'
    print ('dataloader_workers = ',args.workers)
    batch_size=250

    mpi_workers_list = list(range(1, world_size))
    print(mpi_workers_list)
    workers_group = dist.new_group(ranks=mpi_workers_list)
    model = Net().to(device=device)

    for param in model.parameters():
        param.grad = torch.zeros(param.size(), requires_grad=True)
        param.grad.data.zero_()

    if current_rank == server_rank:
        run_server(world_size, batch_size, model)
    else:
        run_mpi_worker(train_data, img_path, image_extension, model, args.workers, current_rank, world_size, workers_group, steps, batch_size)


    