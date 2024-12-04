import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image  
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# initialize parser
parser = argparse.ArgumentParser(description="This parser contains the starting_epoch and learning rate of training")

# adding the arguments
parser.add_argument('--starting_epoch', type=int, required=True, help='epoch we are starting training from,\
                    this is for naming the files that are going to be saved')
parser.add_argument('--learning_rate', type=float, required=True, help='the learning rate used for training the model')
args = parser.parse_args()

# Initialize some constants
CUDA_VISIBLE_DEVICES=7
DEVICE = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 256
INIT_DIM = 32
LATENT_DIM = 5
NUM_EPOCHS = 1000
BATCH_SIZE = 256
LR_RATE = args.learning_rate # original is 3e-4
STARTING_EPOCH = args.starting_epoch
KERNEL_SIZE = 4

# Dataset Loading
data_path = 'data/dataset/train_set' # setting path
# sequence of transformations to be done
transform = transforms.Compose([transforms.Resize((INPUT_DIM, INPUT_DIM)),   # sequence of transformations to be done
                                transforms.Grayscale(num_output_channels=1), # on each image (resize, greyscale,
                                transforms.ToTensor()])                      # convert to tensor)

dataset = datasets.ImageFolder(root=data_path, transform=transform) # read data from folder

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True) # create dataloader object

if STARTING_EPOCH == 0:
    model = VariationalAutoEncoder(init_dim=INIT_DIM, latent_dim=LATENT_DIM, kernel_size=KERNEL_SIZE).to(DEVICE) # initializing model object
else:
    model = VariationalAutoEncoder(init_dim=INIT_DIM, latent_dim=LATENT_DIM, kernel_size=KERNEL_SIZE)
    model.load_state_dict(torch.load(f'models/general_model_epoch_{STARTING_EPOCH}'))
    model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE) # defining optimizer
loss_fn = nn.BCELoss() # define loss function

# Start Training
avg_losses = torch.tensor(data=[]).to(DEVICE)
for epoch in range(STARTING_EPOCH + 1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))
    print(f'Epoch: {epoch}')
    losses = torch.tensor(data=[])
    losses = losses.to(DEVICE)
    for i, (x, _) in loop:
        # forward pass
        x = x.to(DEVICE)
        x_reconstructed, mu, sigma = model(x)
        
        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        
        # backpropagation
        loss = reconstruction_loss + kl_div
        loss = loss.view((1)).to(DEVICE)
        losses = torch.cat((losses, loss), -1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    avg_loss = losses.mean().view((1)).to(DEVICE)
    avg_losses = torch.cat((avg_losses,avg_loss), -1)
    # avg_losses.append(avg_loss)
    print(f'Loss: {avg_loss}')
    
    if epoch % 20 == 0:
        torch.save(model.state_dict(), f'models/general_model_epoch_{epoch}')
        torch.save(avg_losses, f'models/avg_losses_epoch_{epoch}.pt')

torch.save(model.state_dict(), 'models/general_model_final')