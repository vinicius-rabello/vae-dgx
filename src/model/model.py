import torch
from torch import nn
import torch.nn.functional as F

kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling
input_dim = 256 # input shape (input_dim x input_dim)

# input img -> hidden dim -> mean, std -> reparametrization trick -> decoder -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, init_dim=8,latent_dim=3, kernel_size = 4):
        super().__init__()
        
        # encoder
        
        self.enc1 = nn.Conv2d(
            in_channels=1, out_channels=init_dim, kernel_size=kernel_size, # 1st conv layer
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_dim, out_channels=init_dim*2, kernel_size=kernel_size, # 2nd conv layer
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_dim*2, out_channels=init_dim*4, kernel_size=kernel_size, # 3rd conv layer
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_dim*4, out_channels=init_dim*8, kernel_size=kernel_size, # 4th conv layer
            stride=2, padding=0
        )
        
        # fully connected layer
        
        self.fc1 = nn.Linear(init_dim*1800, init_dim*16) # this needs fixing everytime input dim changes, check x.shape for layer before
        self.fc_mu = nn.Linear(init_dim*16, latent_dim)
        self.fc_log_var = nn.Linear(init_dim*16, latent_dim)
        self.fc2 = nn.Linear(latent_dim, init_dim*32)
        
        # decoder
         
        self.dec1 = nn.ConvTranspose2d(
            in_channels=init_dim*32, out_channels=init_dim*32, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_dim*32, out_channels=init_dim*16, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_dim*16, out_channels=init_dim*8, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_dim*8, out_channels=init_dim*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_dim*4, out_channels=init_dim, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec6 = nn.ConvTranspose2d(
            in_channels=init_dim, out_channels=4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec7 = nn.ConvTranspose2d(
            in_channels=4, out_channels=1, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        
        # defining relu
        self.relu = nn.ReLU()
        
        # defining dropout
        self.dropout = nn.Dropout(0.2)
        
    def encode(self,x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        x = self.relu(self.enc4(x))
        x = torch.flatten(x)
        h=self.fc1(x)
        mu, sigma = self.fc_mu(h), self.fc_log_var(h)
        return mu, sigma
    
    def decode(self,z):
        x = self.fc2(z)
        x = torch.reshape(x, ((256,1,1)))
        x=self.relu(self.dec1(x))
        x=self.relu(self.dec2(x))
        x=self.relu(self.dec3(x))
        x=self.relu(self.dec4(x))
        x=self.relu(self.dec5(x))
        x=self.relu(self.dec6(x))
        return torch.sigmoid(self.dec7(x))
    
    def forward(self,x):
        x = self.dropout(x)
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
    
if __name__=='__main__':
    x=torch.randn(1,input_dim,input_dim)
    vae=VariationalAutoEncoder()
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape, mu.shape, sigma.shape)