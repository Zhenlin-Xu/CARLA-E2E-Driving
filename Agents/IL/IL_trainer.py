import os
import h5py
import time
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utils.net import MultiModActor

@hydra.main(config_path=".", config_name="config")
def imitation_learn(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    '''
    Train and test the agent of IL for CARLA E2E-driving.
    '''

    # setup the model, optim, criterion
    model = MultiModActor().cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.train.lr)
    criterion = nn.MSELoss()


    train_path = cfg.train.path
    test_path = cfg.test.path
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    writer = SummaryWriter(log_dir="/home/gav/Desktop/CARLA/Agents/IL/Logs/"+date_str)

    num_epoch = cfg.train.epoch
    num_train_dataset = len(os.listdir(train_path)) # 900
    num_test_dataset  = len(os.listdir(test_path))  # 100


    for epoch in range(num_epoch):
        
        BEGIN_epoch = time.time()
        train_loss = 0
        test_loss = 0
        best_loss = 10
        
        # Train
        print(f"\nTraining in the epoch: {epoch:3d}\n")
        model.train(True)
        for i, filename in enumerate(os.listdir(train_path)):
            f = h5py.File(train_path+filename, 'r')
            BEGIN_batch = time.time()

            rgb = np.array(f["rgb"]).reshape((200,6*100*200))/255.0
            val = np.array(f["value"])
            act = torch.Tensor(np.array(f['target'])).cuda()        
            out, _ = model(np.concatenate((rgb,val),axis=1), None)
            
            optimizer.zero_grad()       # zero the parameter gradients
            loss = criterion(out, act)  # compute the loss 
            loss.backward()             # backpropagate the loss
            optimizer.step()            # adjust parameters based on the calculated gradients
            train_loss += loss.item() 
            END_batch = time.time()
            if i % 50 == 0:
                print('Epoch: {:3d}/{:3d}'.format(epoch, num_epoch),
                    'Iteration: {:3d}/{:3d}'.format(i, num_train_dataset),
                    'Loss: {:.4f}'.format(loss.item()),
                    'Time: {:.4f}'.format(END_batch - BEGIN_batch))
            f.close()

        if best_loss > train_loss/num_train_dataset:
            torch.save(model.state_dict(), "/home/gav/Desktop/CARLA/Agents/IL/Models/" + "model_weights_"+date_str+'.pth')
            # torch.save(model, "./model/model_"+date_str+'.pth')
            best_loss = train_loss/num_train_dataset
        # logging and output for training epoch
        writer.add_scalar('MSE_epoch_loss/train', train_loss/num_train_dataset, epoch)
        print(f"EPOCH: {epoch:3d} Loss/train: {(train_loss/num_train_dataset):.5f}")  #Loss/test: {test_loss/num_test_dataset}")


        # Test
        print(f"\nTesting in the epoch: {epoch:3d}\n.")
        model.eval()
        for i, filename in enumerate(os.listdir(test_path)):

            f = h5py.File(test_path+filename)
            BEGIN_batch = time.time()

            rgb = np.array(f["rgb"]).reshape((200,6*100*200))/255.0
            val = np.array(f["value"])
            act = torch.Tensor(np.array(f['target'])).cuda()        
            out, _ = model(np.concatenate((rgb,val),axis=1), None)
            loss = criterion(out, act)  # compute the loss 
            test_loss += loss.item() 
            END_batch = time.time()
            if i % 20 == 0:
                print('Epoch: {:3d}/{:3d}'.format(epoch, num_epoch),
                    'Iteration: {:3d}/{:3d}'.format(i, num_test_dataset),
                    'Loss: {:.4f}'.format(loss.item()),
                    'Time: {:.4f}'.format(END_batch - BEGIN_batch))
            f.close()
        # logging and output for testing epoch
        writer.add_scalar('MSE_epoch_loss/test', test_loss/num_test_dataset, epoch)
        
        END_epoch = time.time()
        # logging and output for this epoch

        print("\n___________________________________________")
        writer.add_scalars('MSE_epoch_loss_', {
            'train_loss':  train_loss/num_train_dataset, 
            'test_loss' :  test_loss /num_test_dataset  }
            ,epoch)
        print(f"\nEPOCH: {epoch:3d} Loss/train: {(train_loss/num_train_dataset):.5f}") 
        print(f"EPOCH: {epoch:3d} Loss/test: {(test_loss/num_test_dataset):.5f}")
        print(f"EPOCH: {epoch:3d} TIME: {(END_epoch-BEGIN_epoch)}") 
        print("___________________________________________")


if __name__ == "__main__":
    
    BEGIN = time.time()
    imitation_learn()
    END = time.time()
    print(f"Goodbye Sir!, It takes {END-BEGIN} seconds")