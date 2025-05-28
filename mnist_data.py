from torchvision.datasets import MNIST
from torchvision import transforms

train_set = MNIST(root="mnist",          # directory to store data
                  train=True,
                  download=True,         # <- this triggers the download
                  transform=transforms.ToTensor())

