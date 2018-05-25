import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset, VideoDataset1M
from network import R2Plus1DClassifier

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=45, save=True, path="model_data.pth.tar"):
    r"""Trains the model for a fixed number of epochs, using the specified Dataloaders, 
    criterion, optimizer and scheduler. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    
    Args:
        model (Module): The model which is to be trained. Training is an in place operation.
        dataloaders (dict): Dictionary with the dataloaders for each split assigned by key-value pairs
        criterion (callable): Loss function, takes in predictions and label_array, and outputs the loss
        optimizer (Optimizer): Optimizer for the model
        scheduler (_LR_Scheduler): Scheduler for the optimizer
        num_epochs (int, optional): Number of epochs the model is to be trained for in total, including epochs being restored from save files. Defaults to 45.
        save (bool, optional): If True, the model is to be saved. Defaults to True. 
        path (str, optional): The path the model is to be saved to (if being saved), and restored from. Defaults to "model_data.pth.tar". 
    """

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")

        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()


            for inputs, labels in dataloaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    # we're interested in the indices on the max values, not the values themselves
                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)         

            # compute the average loss and accuracy for this epoch, and print
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

    # save the model if save=True
    if save:
        torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': epoch_acc,
        'opt_dict': optimizer.state_dict(),
        }, path)

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")

# initalize the ResNet 18 version of this model
model = R2Plus1DClassifier(num_classes=2, layer_sizes=[2, 2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification

# prepare the dataloaders into a dict
train_dataloader = DataLoader(VideoDataset('/home/irhum/data/video'), batch_size=32, shuffle=True, num_workers=4)
# IF training on Kinetics-600 and require exactly a million samples each epoch, 
# import VideoDataset1M and uncomment the following
# train_dataloader = DataLoader(VideoDataset1M('/home/irhum/data/video'), batch_size=32, num_workers=4)
val_dataloader = DataLoader(VideoDataset('/home/irhum/data/video', mode='val'), batch_size=32, num_workers=4)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

# hyperparameters as given in paper sec 4.1
optimizer = optim.SGD(model.parameters(), lr=0.01)
# the scheduler divides the lr by 10 every 10 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=45)
