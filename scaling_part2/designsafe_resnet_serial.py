import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline

import os


def load_datasets(train_path, val_path, test_path):
  val_img_transform = transforms.Compose([transforms.Resize((244,244)),
                                         transforms.ToTensor()])
  #  Main Modification: Additional transformation
  train_img_transform = 


  train_dataset = datasets.ImageFolder(train_path, transform=train_img_transform)
  val_dataset = datasets.ImageFolder(val_path, transform=val_img_transform) 
  test_dataset = datasets.ImageFolder(test_path, transform=val_img_transform) if test_path is not None else None
  print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
  return train_dataset, val_dataset, test_dataset

def construct_dataloaders(train_set, val_set, test_set, batch_size, shuffle=True):
  train_dataloader = torch.utils.data.DataLoader(train_set, batch_size, shuffle)
  val_dataloader = torch.utils.data.DataLoader(val_set, batch_size) 
  test_dataloader = torch.utils.data.DataLoader(test_set, batch_size) if test_path is not None else None
  return train_dataloader, val_dataloader, test_dataloader


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

def load_checkpoint(checkpoint_path, DEVICE):
  checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
  return checkpoint

@torch.no_grad()
def eval_model(data_loader, model, loss_fn, DEVICE):
  model.eval()
  loss, accuracy = 0.0, 0.0
  n = len(data_loader)

  for i, data in enumerate(data_loader):
    x,y = data
    x,y = x.to(DEVICE), y.to(DEVICE)
    pred = model(x)
    loss += loss_fn(pred, y)/len(x)
    pred_label = torch.argmax(pred, axis = 1)
    accuracy += torch.sum(pred_label == y)/len(x)

  return loss/n, accuracy/n 

def train(train_loader, val_loader, model, opt, scheduler, loss_fn, epochs, DEVICE, checkpoint_file, prev_best_val_acc):
  n = len(train_loader)
  
  best_val_acc = torch.tensor(0.0).cuda() if prev_best_val_acc is None else prev_best_val_acc
    
  for epoch in range(epochs):
    model.train(True)
    
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
    
    #######################################
    # Learning rate reducer takes action ##
    #######################################
    
    

    avg_loss, avg_acc = avg_loss/n, avg_acc/n
    
    #########################################################
    # Save the best model that has the highest val accuracy #
    #########################################################
    if val_acc.item() > best_val_acc.item():
      print(f"\nPrev Best Val Acc: {best_val_acc} < Cur Val Acc: {val_acc}")
      print("Saving the new best model...")
      
      #save the model with torch.save
      torch.save({})
    
      best_val_acc = val_acc
      print("Finished saving model\n")
        
    # Print the metrics (should be same on all machines)
    print(f"\n(Epoch {epoch+1}/{epochs}) Time: {total_time}s")
    print(f"(Epoch {epoch+1}/{epochs}) Average train loss: {avg_loss}, Average train accuracy: {avg_acc}")
    print(f"(Epoch {epoch+1}/{epochs}) Val loss: {val_loss}, Val accuracy: {val_acc}")  
    print(f"(Epoch {epoch+1}/{epochs}) Current best val acc: {best_val_acc}\n")  

if __name__ == "__main__":
    torch.hub.set_dir('/tmp') # remove when not running here 
    hp = {"lr":1e-4, "batch_size":16, "epochs":5}
    train_path, val_path,test_path = "/tmp/Dataset_2/Train/", "/tmp/Dataset_2/Validation/", None 
    train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(train_set, val_set, test_set, hp["batch_size"], True)
    resnet = getResNet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device

    resnet.to(device)
    
    # For saving the trained model
    model_folder_path = os.getcwd()+"/output_model/"
    os.makedirs(model_folder_path,exist_ok=True)

    # filename for our best model
    checkpoint_file = model_folder_path+"best_model.pt"

    # load the checkpoint that has the best performance in previous experiments
    prev_best_val_acc = None
    checkpoint_file = model_folder_path+"best_model.pt"
    if os.path.exists(checkpoint_file):
        checkpoint = load_checkpoint(checkpoint_file, device)
        prev_best_val_acc = checkpoint['accuracy']
 
