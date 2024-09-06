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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_DIM = 256
INIT_DIM = 8
LATENT_DIM = 3
NUM_EPOCHS = 3000
BATCH_SIZE = 1
LR_RATE = 3e-4
KERNEL_SIZE = 4

# Dataset Loading
data_path = 'data/dataset/train_set' # setting path
# sequence of transformations to be done
transform = transforms.Compose([transforms.Resize((INPUT_DIM, INPUT_DIM)),   # sequence of transformations to be done
                                transforms.Grayscale(num_output_channels=1), # on each image (resize, greyscale,
                                transforms.ToTensor()])                      # convert to tensor)

dataset = datasets.ImageFolder(root=data_path, transform=transform) # read data from folder

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True) # create dataloader object

model = VariationalAutoEncoder(init_dim=INIT_DIM, latent_dim=LATENT_DIM, kernel_size=KERNEL_SIZE).to(DEVICE) # initializing model object
# model = VariationalAutoEncoder(init_dim=INIT_DIM, latent_dim=LATENT_DIM, kernel_size=KERNEL_SIZE)
# model.load_state_dict(torch.load('models/general_model_epoch_600'))
# model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE) # defining optimizer
loss_fn = nn.BCELoss(reduction='sum') # define loss function

# Start Training
for epoch in range(1, NUM_EPOCHS + 1):
    loop = tqdm(enumerate(train_loader))
    print(f'Epoch: {epoch}')
    losses = []
    avg_losses = []
    for i, (x, _) in loop:
        # forward pass
        x = x.to(DEVICE).view(1, INPUT_DIM, INPUT_DIM)
        x_reconstructed, mu, sigma = model(x)
        
        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed, x)
        kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        
        # backpropagation
        loss = reconstruction_loss + kl_div
        losses.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    avg_loss = np.mean(losses)
    avg_losses.append(avg_loss)
    print(f'Loss: {avg_loss}')

    # display images
    x = x[0].numpy()
    x_reconstructed = x_reconstructed[0].detach().numpy()
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(x, cmap='gray')
    ax2.imshow(x_reconstructed, cmap='gray')
    plt.show()
    
    if epoch % 50 == 0:
        torch.save(model.state_dict(), f'models/general_model_epoch_{epoch}')

torch.save(model.state_dict(), 'models/general_model_final')