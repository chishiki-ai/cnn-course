import torch.distributed as dist
import torch 
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

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
                                  shuffle=True,
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

def main(device):
    train_dataloader = prepare_data()

    model = get_model().to(device)
    
    # instantiate loss and optimizer 
    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # Train Model 
    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(device, train_dataloader, model, loss_fn, optimizer)

    print("Done!")
    return model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(device)
