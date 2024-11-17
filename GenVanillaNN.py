import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ", Skeleton.full_dim, ")")

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        # Preprocess skeleton (input)
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # Preprocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten())
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0], 1, 1)
        return ske

    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        return denormalized_image


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GenNNSkeToImage(nn.Module):
    """ Class that generates a new image from videoSke from a new skeleton posture.
        Function generator(Skeleton) -> Image
    """
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),  # Output: 64x64 image
            nn.Tanh()
        )
        self.model.apply(init_weights)

    def forward(self, z):
        img = self.model(z)
        return img



class GenVanillaNN:
    """ Class that generates a new image from videoSke from a new skeleton posture.
        Function generator(Skeleton) -> Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.netG = GenNNSkeToImage()
        src_transform = None
        self.filename = 'data/DanceGenVanillaFromSke.pth'

        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG.load_state_dict(torch.load(self.filename))

    def train(self, n_epochs=20):
        # Set the model to training mode
        self.netG.train()

        # Define loss function (Mean Squared Error Loss) for comparing generated images with real images
        criterion = nn.MSELoss()

        # Define the optimizer (Adam) to update the generator's weights
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.001)

        # Loop over the number of epochs
        for epoch in range(n_epochs):
            epoch_loss = 0.0

            # Iterate over batches from the data loader
            for batch_idx, (skeletons, real_images) in enumerate(self.dataloader):
                optimizer.zero_grad()  # Reset gradients for the optimizer

                # Generate images from the skeletons
                generated_images = self.netG(skeletons)

                # Compute the loss between generated images and real images
                loss = criterion(generated_images, real_images)

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

                # Accumulate the batch loss
                epoch_loss += loss.item()

            # Print the average loss for this epoch
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss / len(self.dataloader)}')

        # After training, save the trained generator model
        torch.save(self.netG.state_dict(), self.filename)
        print(f'Model saved to {self.filename}')

    def generate(self, ske):
        """ Generator of image from skeleton """
        self.netG.eval()  # Set the network to evaluation mode

        # Preprocess the input skeleton to the format expected by the network
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)  # Add batch dimension (1, 26, 1, 1)

        with torch.no_grad():  # Disable gradient computation for inference
            # Generate the normalized image from the skeleton
            normalized_output = self.netG(ske_t_batch)

        # Convert the normalized output (Tensor) back to a displayable image (numpy array)
        res = self.dataset.tensor2image(normalized_output[0])  # Get the first image from the batch

        return res


if __name__ == '__main__':
    force = False
    optSkeOrImage = 2  # Use as input a skeleton (1) or an image with a skeleton drawn (2)
    n_epoch = 200  # 200

    train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)  # Load from file

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        nouvelle_taille = (256, 256)
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)

