import numpy as np
import cv2
import os
import sys
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch.optim as optim

from VideoSkeleton import VideoSkeleton  
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
    def __init__(self, videoSke, ske_reduced, image_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]
        image = self.image_transform(ske)  
        
        # Convert image to tensor and normalize
        image = transforms.ToTensor()(image)  
        image = image.float()  
        image = (image - 0.5) / 0.5  

        target_image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            target_image = self.target_transform(target_image)

        return image, target_image  

    def tensor2image(self, tensor):
        """ Convert tensor back to image format. """
        # Detach from the computation graph and denormalize
        tensor = (tensor.detach() + 1) / 2  
        tensor = tensor.clamp(0, 1)  
        image = tensor.permute(1, 2, 0).cpu().numpy()  
        image = (image * 255).astype(np.uint8)  
        return image

class GenNNSkeToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),     
            nn.Tanh()  
        )

    def forward(self, img):
        return self.model(img)  

class GenVanillaNN2():
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        self.netG = GenNNSkeToImage()
        self.filename = 'data/DanceGenVanillaFromSke2.pth'

        ske_transform = SkeToImageTransform(image_size)  
        tgt_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
        ])

        # Pass ske_transform to VideoSkeletonDataset
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, image_transform=ske_transform, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)

        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            self.netG.load_state_dict(torch.load(self.filename))

    def train(self, n_epochs=20):
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # Mean Squared Error for image generation
        self.netG.train()

        for epoch in range(n_epochs):
            for ske_image, target_img in self.dataloader:
                optimizer.zero_grad()
                output = self.netG(ske_image)
                loss = criterion(output, target_img)
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

        torch.save(self.netG.state_dict(), self.filename)  # Save the model after training

        print('Finished Training')

    def generate(self, ske):
        """ Generator of image from skeleton image """
        image = self.dataset.image_transform(ske)  # Generate an image from the skeleton
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)  
        normalized_output = self.netG(image_tensor)
        res = self.dataset.tensor2image(normalized_output[0])  
        return res

if __name__ == '__main__':
    force = False
    n_epoch = 200 
    train = True  

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"

    targetVideoSke = VideoSkeleton(filename)

    if train:
        gen = GenVanillaNN2(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN2(targetVideoSke, loadFromFile=True)  

    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        image = cv2.resize(image, (256, 256))
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
        if key == 27:  # Esc key to exit
            break

    cv2.destroyAllWindows()  
