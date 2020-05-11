import os
import time
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from metric import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Training on {}'.format(device))

latent_dim = 64
hidden_dim = 256
input_dim = 784
num_epochs = 200
batch_size = 100
model_dir = 'outputs_gan'
sample_dir = model_dir+'/samples'

if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5],std=[0.5])])

mnist = torchvision.datasets.MNIST(root='../data/',train=True,transform=transform,download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist,batch_size=batch_size,shuffle=True)

D = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_dim, 1),
    nn.Sigmoid()
).to(device)

G = nn.Sequential(
    nn.Linear(latent_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, input_dim),
    nn.Tanh()
).to(device)

print(D)
print(G)

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

def denorm(x):
	    out = (x + 1) / 2
	    return out.clamp(0, 1)
	
def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

steps = len(data_loader)
start = time.time()
d_losses = []
g_losses = []
metrics = []
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # Training the Discriminator
        outputs = D(images)
        d_loss_real = loss_fn(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = loss_fn(outputs, fake_labels)
        fake_score = outputs
        
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # Training the Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        g_loss = loss_fn(outputs, real_labels)
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        metrics.append(compute_score(images.data, fake_images.data  ))

        if (i+1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}' 
	                  .format(epoch, num_epochs, i+1, steps, d_loss.item(), g_loss.item()))
    # Save real images
    if (epoch+1) == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # Save sampled images
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)

    save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

end = time.time()
print('Total training time: {}'.format(end - start))

import pickle
with open('./'+model_dir+'/d_losses.pkl', "wb") as f:
    pickle.dump(d_losses, f)
with open('./'+model_dir+'/g_losses.pkl', "wb") as f:
    pickle.dump(g_losses, f)
with open('./'+model_dir+'/metrics.pkl', "wb") as f:
    pickle.dump(metrics, f)
