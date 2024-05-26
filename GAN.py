"""from att_unet_2d import att_unet_2d
from discriminator_2d import discriminator_2d
from losses_for_gan import huber_reverse_loss, loss_D, l1_loss
import torch
import torch.nn as nn
import torch.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
])
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
image, label = train_dataset[0]
print(image.shape)
test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=4)
filters = [64, 128, 256, 512]
model_G = att_unet_2d(input_size=image.shape, filter_num=filters, n_labels=1)

model_D = discriminator_2d(input_size=image.shape, filter_num=filters,stack_num_down=2, activation='ReLU', batch_norm=False, pool = False, backbone=None)
    

def D_train_step(real_images, fake_images, model_D, D_optimizer):
    model_D.train()
    
    D_optimizer.zero_grad()
    
    D_real_output = model_D(real_images)
    D_fake_output = model_D(fake_images.detach())  # Detach to avoid backprop through generator
    
    D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
    
    D_total_loss.backward()
    D_optimizer.step()
    
    return D_total_loss.item(), D_real_loss.item(), D_fake_loss.item()

epoch_num = 10
def G_train_step(input_image, target, model_G, model_D, G_optimizer):
    model_G.train()
    model_D.eval()  # Discriminator is not updated during generator training step
    
    G_optimizer.zero_grad()
    
    G_outputs = model_G(input_image)
    D_fake_output = model_D(G_outputs)
    
    loss_hubber = huber_reverse_loss(G_outputs, target)
    loss_dis = torch.mean((1-D_fake_output)**2)
    G_total_loss = loss_hubber+loss_dis
    G_total_loss.backward()
    G_optimizer.step()
    return G_total_loss.item(), loss_hubber.item(), loss_dis.item()


G_optimizer = torch.optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
for _ in range(epoch_num):
    for i, data in enumerate(train_loader):
        image, label = data
        
        G_loss, G_loss_hubber, G_loss_dis = G_train_step(image, , epoch, model_G, model_D, G_optimizer, train_config)
        """
"""import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from att_unet_2d import att_unet_2d
from discriminator_2d import discriminator_2d
from losses_for_gan import huber_reverse_loss, loss_D

# Define dataset class to pair source and target digit images
class MNISTPairDataset(Dataset):
    def __init__(self, mnist_dataset, source_digit, target_digit, transform=None):
        self.mnist_dataset = mnist_dataset
        self.source_digit = source_digit
        self.target_digit = target_digit
        self.transform = transform
        
        self.source_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == self.source_digit]
        self.target_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == self.target_digit]
        
    def __len__(self):
        return min(len(self.source_indices), len(self.target_indices))
    
    def __getitem__(self, idx):
        source_idx = self.source_indices[idx % len(self.source_indices)]
        target_idx = self.target_indices[idx % len(self.target_indices)]
        
        source_image, _ = self.mnist_dataset[source_idx]
        target_image, _ = self.mnist_dataset[target_idx]
        
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        return source_image, target_image

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
])

def main():
    # Load MNIST dataset
    train_dataset = MNIST(root='./data', train=True, download=True)
    test_dataset = MNIST(root='./data', train=False, download=True)

    # Create paired datasets
    source_digit = 4
    target_digit = 9
    train_subset = MNISTPairDataset(train_dataset, source_digit=source_digit, target_digit=target_digit, transform=transform)
    test_subset = MNISTPairDataset(test_dataset, source_digit=source_digit, target_digit=target_digit, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize the models
    filters = [64, 128, 256, 512]
    model_G = att_unet_2d(input_size=(1, 28, 28), filter_num=filters, n_labels=1)
    model_D = discriminator_2d(input_size=(1, 28, 28), filter_num=filters, stack_num_down=2, activation='ReLU', batch_norm=False, pool=False, backbone=None)

    # Define optimizers
    G_optimizer = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    epoch_num = 10

    for epoch in range(epoch_num):
        for i, (input_image, target) in enumerate(train_loader):
            # Discriminator training step
            model_D.train()
            D_optimizer.zero_grad()
            D_real_output = model_D(target)
            G_outputs = model_G(input_image)
            D_fake_output = model_D(G_outputs.detach())  # Detach to avoid backprop through generator
            D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
            D_total_loss.backward()
            D_optimizer.step()

            # Generator training step
            model_G.train()
            model_D.eval()  # Discriminator is not updated during generator training step
            G_optimizer.zero_grad()
            G_outputs = model_G(input_image)
            D_fake_output = model_D(G_outputs)
            loss_hubber = huber_reverse_loss(G_outputs, target)
            loss_dis = torch.mean((1 - D_fake_output) ** 2)
            G_total_loss = loss_hubber + loss_dis
            G_total_loss.backward()
            G_optimizer.step()

            # Print losses
            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epoch_num}], Step [{i}/{len(train_loader)}], "
                      f"D Loss: {D_total_loss:.4f}, G Loss: {G_total_loss:.4f}, "
                      f"D Real Loss: {D_real_loss:.4f}, D Fake Loss: {D_fake_loss:.4f}, "
                      f"G Hubber Loss: {loss_hubber:.4f}, G Dis Loss: {loss_dis:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from att_unet_2d import att_unet_2d
from discriminator_2d import discriminator_2d
from losses_for_gan import huber_reverse_loss, loss_D

# Define dataset class to pair source and target digit images
class MNISTPairDataset(Dataset):
    def __init__(self, mnist_dataset, source_digit, target_digit, transform=None):
        self.mnist_dataset = mnist_dataset
        self.source_digit = source_digit
        self.target_digit = target_digit
        self.transform = transform
        
        self.source_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == self.source_digit]
        self.target_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == self.target_digit]
        
    def __len__(self):
        return min(len(self.source_indices), len(self.target_indices))
    
    def __getitem__(self, idx):
        source_idx = self.source_indices[idx % len(self.source_indices)]
        target_idx = self.target_indices[idx % len(self.target_indices)]
        
        source_image, _ = self.mnist_dataset[source_idx]
        target_image, _ = self.mnist_dataset[target_idx]
        
        if self.transform:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        
        return source_image, target_image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
])

def main():
    # Load MNIST dataset
    train_dataset = MNIST(root='./data', train=True, download=True)
    test_dataset = MNIST(root='./data', train=False, download=True)

    # Create paired datasets
    source_digit = 4
    target_digit = 9
    train_subset = MNISTPairDataset(train_dataset, source_digit=source_digit, target_digit=target_digit, transform=transform)
    test_subset = MNISTPairDataset(test_dataset, source_digit=source_digit, target_digit=target_digit, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize the models
    filters = [64, 128, 256, 512]
    model_G = att_unet_2d(input_size=(1, 64, 64), filter_num=filters, n_labels=1)
    model_D = discriminator_2d(input_size=(1, 64, 64), filter_num=filters, stack_num_down=2, activation='ReLU', batch_norm=False, pool=False, backbone=None)

    # Define optimizers
    G_optimizer = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    epoch_num = 10

    for epoch in range(epoch_num):
        for i, (input_image, target) in enumerate(train_loader):
            # Discriminator training step
            model_D.train()
            D_optimizer.zero_grad()
            D_real_output = model_D(target)
            G_outputs = model_G(input_image)
            D_fake_output = model_D(G_outputs.detach())  # Detach to avoid backprop through generator
            D_total_loss, D_real_loss, D_fake_loss = loss_D(D_real_output, D_fake_output)
            D_total_loss.backward()
            D_optimizer.step()

            # Generator training step
            model_G.train()
            model_D.eval()  # Discriminator is not updated during generator training step
            G_optimizer.zero_grad()
            G_outputs = model_G(input_image)
            D_fake_output = model_D(G_outputs)
            loss_hubber = huber_reverse_loss(G_outputs, target)
            loss_dis = torch.mean((1 - D_fake_output) ** 2)
            G_total_loss = loss_hubber + loss_dis
            G_total_loss.backward()
            G_optimizer.step()

            # Print losses
            if i % 10 == 0:
                print(f"Epoch [{epoch}/{epoch_num}], Step [{i}/{len(train_loader)}], "
                      f"D Loss: {D_total_loss:.4f}, G Loss: {G_total_loss:.4f}, "
                      f"D Real Loss: {D_real_loss:.4f}, D Fake Loss: {D_fake_loss:.4f}, "
                      f"G Hubber Loss: {loss_hubber:.4f}, G Dis Loss: {loss_dis:.4f}")

    print("Training complete.")

if __name__ == "__main__":
    main()
