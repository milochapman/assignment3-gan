import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib.model import GANGenerator, GANDiscriminator
from helper_lib.trainer import train_gan

BATCH_SIZE = 128
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

gen = GANGenerator(z_dim=100)
disc = GANDiscriminator()

train_gan(gen, disc, loader, z_dim=100, device=DEVICE, epochs=EPOCHS)

torch.save(gen.state_dict(), "gan_generator_mnist.pth")
torch.save(disc.state_dict(), "gan_discriminator_mnist.pth")
