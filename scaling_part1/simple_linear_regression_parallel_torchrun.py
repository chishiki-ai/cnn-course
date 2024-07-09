import torch.distributed as dist
import torch 
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


##############################################
# 1. Create a process group
def init_distributed():
    '''
    local_rank: identifier for pariticular GPU on one node
    world: total number of process in a the group
    '''
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)                       #
    dist.init_process_group("nccl",                   # backend being used; nccl typically used with distributed GPU training
                            rank=local_rank,                # rank of the current process being used
                            world_size=world_size)    # total number of processors being used
##############################################

def get_model():
   return torch.nn.Sequential(
            torch.nn.Linear(1, 1),     # first number specifies input dimension; second number specifies output dimension
            ) 

def prepare_data(batch_size=32):
    # Generate random data centered around 10 with noise
    X = torch.randn(32*4, 1) * 10
    y = X + torch.randn(32*4, 1) * 3
    
    # pass data to the distributed sampler and dataloader 
    train_dataloader = DataLoader(list(zip(X,y)),
                                  ##############################################
                                  # 2. Use Pytorch's DistributedSampler to ensure that data passed to each GPU is different
                                  shuffle=False,
                                  sampler=DistributedSampler(list(zip(X,y))),
                                  ##############################################
                                  batch_size=batch_size)
    
    return train_dataloader

# training loop for one epoch
def train_loop(rank, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # transfer data to GPU if available
        X = X.to(rank)
        y = y.to(rank)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 2 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def main():
    local_rank = int(os.environ['LOCAL_RANK'])

    init_distributed()

    train_dataloader = prepare_data()

    ##############################################
    # 3. Wrap Model with Pytorch's DistributedDataParallel
    model = DDP(get_model().to(local_rank), device_ids=[local_rank], output_device=local_rank)
    ##############################################
    
    # instantiate loss and optimizer 
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Train Model 
    epochs = 20
    for t in range(epochs):
        ################################################
        # 4. Only write/print model information on one GPU
        if local_rank == 0:
            print(f"Epoch {t+1}\n-------------------------------")
        ################################################
        train_loop(local_rank, train_dataloader, model, loss_fn, optimizer)

    #################################################
    # 5. Close Process Group
    dist.destroy_process_group()
    #################################################

    print("Done!")
    return model

if __name__ == "__main__":
    main()
