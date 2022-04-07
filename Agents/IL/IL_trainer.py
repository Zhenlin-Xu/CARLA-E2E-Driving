import os
from re import I
import h5py
import time
import datetime

import numpy as np
from tomlkit import key
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from utils.net import MultiModActor

model = MultiModActor().cuda()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
criterion = nn.MSELoss()

BEGIN = time.time()

dataset_path = "/media/gav/Gavin/User/ubuntu/DATASETS/2022ImitationLearningData"
file = "Town01_Opt_normal.hdf5"

# dataset_path = '/home/gav/Desktop/CARLA/Agents/IL/Datasets'
# file = 'Town01_normal.hdf5'

f = h5py.File(os.path.join(dataset_path, file), "r")

date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
writer = SummaryWriter(log_dir="./Agents/IL/Logs/"+date_str)

num_epoch = 10
num_train_dataset = 500 # 500

model.train(True)

for epoch in range(num_epoch):
    
    print()
    train_loss = 0
    test_loss = 0
    best_loss = 10
    

    # for i, idx in enumerate(list(f.keys())[:num_train_dataset]):
    for i, idx in enumerate(list(f.keys())):
        # print(f[idx][:,:-2].shape)
        BEGIN_ = time.time()
        optimizer.zero_grad()       # zero the parameter gradients

        act = torch.Tensor(f[idx][:,-2:]).cuda()
        # print(act.shape)
        out, _ = model(f[idx][:,:-2], None)
        # print(out.shape)

        loss = criterion(out, act)  # compute the loss 
        loss.backward()             # backpropagate the loss
        optimizer.step()            # adjust parameters based on the calculated gradients
        train_loss += loss.item() 
        END_ = time.time()
        if i % 25 == 0:
            print('Epoch: {}/{}'.format(epoch, num_epoch),
                  'Iteration: {}/{}'.format(i, num_train_dataset),
                  'Loss: {:.4f}'.format(loss.item()),
                  'Time: {:.4f}'.format(END_ - BEGIN_))

        writer.add_scalar('MSE_batch_loss/train', loss.item(), epoch*num_train_dataset+i)

    if best_loss > train_loss/num_train_dataset:
        torch.save(model.state_dict(), "./Agents/IL/Models/" + "model_weights_"+date_str+'.pth')
        # torch.save(model, "./model/model_"+date_str+'.pth')
        best_loss = train_loss/num_train_dataset

    writer.add_scalar('MSE_epoch_loss/train', train_loss/num_train_dataset, epoch)
    print(f"EPOCH: {epoch:3d} Loss/train: {(train_loss/num_train_dataset):.5f} TIME: {(END_-BEGIN_)}")  #Loss/test: {test_loss/num_test_dataset}")

f.close()
END = time.time()
print(END-BEGIN)

print("Goodbye Sir!")