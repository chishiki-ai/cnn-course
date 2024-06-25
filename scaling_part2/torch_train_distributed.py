# NOTE: This is the main script of using Distributed Data Parallel to train ResNet


import sys
import os
import numpy as np
import gc

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from datetime import datetime
import warnings
import shutil

warnings.filterwarnings("ignore", message="torch.distributed._all_gather_base is a private function and will be deprecated. Please use torch.distributed.all_gather_into_tensor instead.")

# Define the GPUs that will be used in this script
os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(str(x) for x in list(range(torch.cuda.device_count())))

# Apply transformations to our data.
# The datasets transformations are the same as the ones from part 2 of this tutorial.
def load_datasets(train_path, val_path, test_path):
  val_img_transform = transforms.Compose([transforms.Resize((244,244)),
                                         transforms.ToTensor()])
  train_img_transform = transforms.Compose([transforms.AutoAugment(),
                                           transforms.Resize((244,244)),
                                           transforms.ToTensor()])
  train_dataset = datasets.ImageFolder(train_path, transform=train_img_transform)
  val_dataset = datasets.ImageFolder(val_path, transform=val_img_transform) 
  test_dataset = datasets.ImageFolder(test_path, transform=val_img_transform) if test_path is not None else None
  
  local_rank = int(os.environ['LOCAL_RANK'])
  if local_rank == 0:
    print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
  return train_dataset, val_dataset, test_dataset

# Construct Dataloaders
# The DistributedSampler we use here restricts data loading to a subset of the dataset.
# In conjunction with DistributedDataParallel (shows up later in this tutorial), each process can pass a DistributedSampler instance as a DataLoader sampler, and load a subset of the original dataset that is exclusive to it.
def construct_dataloaders(train_set, val_set, test_set, batch_size, shuffle=True):  
  train_sampler = DistributedSampler(dataset=train_set,shuffle=shuffle)
  val_sampler = DistributedSampler(dataset=val_set, shuffle=False)
  test_sampler = DistributedSampler(dataset=test_set, shuffle=False) if test_set is not None else None
  
  train_dataloader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,sampler=train_sampler,num_workers=4,pin_memory=True)
  val_dataloader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,sampler=val_sampler,num_workers=4)
  test_dataloader = torch.utils.data.DataLoader(test_aset, batch_size, sampler=test_sampler,num_workers=4) if test_set is not None else None
    
  return train_dataloader, val_dataloader, test_dataloader


# Building the Neural Network
# This is the same from part 2 of this tutorial
def getResNet():
  resnet = models.resnet34(weights='IMAGENET1K_V1')

  # Fix the conv layers parameters
  for conv_param in resnet.parameters():
    conv_param.require_grad = False

  # get the input dimension for this layer
  num_ftrs = resnet.fc.in_features
    
  # build the new final mlp layers of network
  fc = nn.Sequential(
          nn.Linear(num_ftrs, num_ftrs),
          nn.ReLU(),
          nn.Linear(num_ftrs, 3)
        )
    
  # replace final fully connected layer
  resnet.fc = fc
  return resnet


# Model evaluation.
# This is implemented in the same way as part 2 of this tutorial.
@torch.no_grad()
def eval_model(data_loader, model, loss_fn, DEVICE):
  model.train(False)
  model.eval()
  loss, accuracy = 0.0, 0.0
  n = len(data_loader)
    
  local_rank = int(os.environ['LOCAL_RANK'])

  for i, data in enumerate(data_loader):
    x,y = data
    x,y = x.to(DEVICE), y.to(DEVICE)
    pred = model(x)
    loss += loss_fn(pred, y)/len(x)
    pred_label = torch.argmax(pred, axis = 1)
    accuracy += torch.sum(pred_label == y)/len(x)
    
  return loss/n, accuracy/n

# Model training.
# This is implemented in the same way as part 2 of this tutorial.
def train(train_loader, val_loader, model, opt, scheduler, loss_fn, epoch_start, epochs, DEVICE, checkpoint_file, prev_best_val_acc):
  n = len(train_loader)

  local_rank = int(os.environ['LOCAL_RANK'])
  
  best_val_acc = torch.tensor(0.0).cuda() if prev_best_val_acc is None else prev_best_val_acc
    
  for epoch in range(epoch_start, epochs):
    model.train(True)
    
    train_loader.sampler.set_epoch(epoch)
    
    avg_loss, val_loss, val_acc, avg_acc  = 0.0, 0.0, 0.0, 0.0
    
    start_time = datetime.now()
    
    for x, y in train_loader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      pred = model(x)
      loss = loss_fn(pred,y)

      opt.zero_grad()
      loss.backward()
      opt.step()

      avg_loss += loss.item()/len(x)
      pred_label = torch.argmax(pred, axis=1)
      avg_acc += torch.sum(pred_label == y)/len(x)

    val_loss, val_acc = eval_model(val_loader, model, loss_fn, DEVICE)
    
    end_time = datetime.now()
    
    total_time = torch.tensor((end_time-start_time).seconds).cuda()
    
    # Learning rate reducer takes action
    scheduler.step(val_loss)
    
    avg_loss, avg_acc = avg_loss/n, avg_acc/n
    
    # Only machine rank==0 (master machine) saves the model and prints the metrics    
    if local_rank == 0:
        
      # Save the best model that has the highest val accuracy
      if val_acc.item() > best_val_acc.item():
        print(f"\nPrev Best Val Acc: {best_val_acc} < Cur Val Acc: {val_acc}")
        print("Saving the new best model...")
        torch.save({
                'epoch':epoch,
                'machine':local_rank,
                'model_state_dict':model.module.state_dict(),
                'accuracy':val_acc,
                'loss':val_loss
        }, checkpoint_file)
        best_val_acc = val_acc
        print("Finished saving model\n")
        
      # Print the metrics (should be same on all machines)
      print(f"\n(Epoch {epoch+1}/{epochs}) Time: {total_time}s")
      print(f"(Epoch {epoch+1}/{epochs}) Average train loss: {avg_loss}, Average train accuracy: {avg_acc}")
      print(f"(Epoch {epoch+1}/{epochs}) Val loss: {val_loss}, Val accuracy: {val_acc}")  
      print(f"(Epoch {epoch+1}/{epochs}) Current best val acc: {best_val_acc}\n")  

def load_checkpoint(checkpoint_path, DEVICE):
  checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
  return checkpoint

def load_model_fm_checkpoint(checkpoint, primitive_model):
  primitive_model.load_state_dict(checkpoint['model_state_dict'])
  return primitive_model 

def init_distributed():
    
  dist_url = "env://"
  
  world_size = int(os.environ['WORLD_SIZE']) 
  local_rank = int(os.environ['LOCAL_RANK']) 
  dist.init_process_group(backend="nccl", #"nccl" for using GPUs, "gloo" for using CPUs
                          init_method=dist_url, 
                          world_size=world_size, 
                          rank=local_rank)
  torch.cuda.set_device(local_rank)


    
def cleanup():
  print("Cleaning up the distributed environment...")
  dist.destroy_process_group()
  print("Distributed environment has been properly closed")
    
    
def main():
  hp = {"lr":1e-4, "batch_size":16, "epochs":5}
  # Please specify the path to train, cross_validation, and test images below:
  train_path, val_path, test_path = "/tmp/Dataset_2/Train/", "/tmp/Dataset_2/Validation/", None
  local_rank = int(os.environ['LOCAL_RANK'])
  DEVICE = torch.device("cuda", local_rank)
    
  # For saving the trained model
  model_folder_path = os.getcwd()+"/output_model/"
  os.makedirs(model_folder_path,exist_ok=True)
    
  # same loss function as part 2 
  loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).cuda()
  train_set, val_set, test_set = load_datasets(train_path, val_path, test_path)
  train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(train_set, val_set, test_set, hp["batch_size"], True)
                          
  model = getResNet().to(DEVICE)
  
    
  # load the checkpoint that has the best performance in previous experiments
  prev_best_val_acc = None
  checkpoint_file = model_folder_path+"best_model.pt"
  if os.path.exists(checkpoint_file):
    checkpoint = load_checkpoint(checkpoint_file, DEVICE)
    prev_best_val_acc = checkpoint['accuracy']
    model = load_model_fm_checkpoint(checkpoint,model)
    epoch_start = checkpoint['epoch']
    if local_rank == 0:
      print(f"resuming training from epoch {epoch_start}")
  else:
    epoch_start = 0
  
  model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
  model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
  opt = torch.optim.Adam(model.parameters(),lr=hp["lr"])
    
  # same learning rate scheduler as part 2
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',factor=0.1, patience=5, min_lr=1e-8, verbose=True)
  
  train(train_dataloader, val_dataloader, model, opt, scheduler, loss_fn, epoch_start, hp["epochs"], DEVICE, checkpoint_file, prev_best_val_acc)
   

  # only the node with rank 0 does the loading, evaluation and printing to avoild duplicate 
  if local_rank == 0:
    primitive_model = getResNet().to(DEVICE)
    checkpoint = load_checkpoint(checkpoint_file, DEVICE)
    best_model = load_model_fm_checkpoint(checkpoint,primitive_model)
    loss, acc = eval_model(val_dataloader,best_model,loss_fn,DEVICE)
    print(f"\nBest model (val loss: {loss}, val accuracy: {acc}) has been saved to {checkpoint_file}\n")
    cleanup()

if __name__ == '__main__':
  init_distributed()
    
  gc.collect()
  for i in range(torch.cuda.device_count()):
    with torch.cuda.device(f"cuda:{i}"):
      torch.cuda.empty_cache()
  
  main()



