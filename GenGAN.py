
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 

import torch.optim as optim

class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()  # Sigmoid to output probability of real/fake
        )


    def forward(self, input):
        return self.main(input).view(-1)  
        #return self.model(input)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.0
        self.fake_label = 0.0
        self.filename = 'data/Dance/DanceGenGAN.pth'
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG.to(device)
        self.netD.to(device)

        # Loss and Optimizers
        criterion = nn.BCELoss()
        optimizerD = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        for epoch in range(n_epochs):
            for i, (skeletons, real_images) in enumerate(self.dataloader):
                skeletons = skeletons.to(device)
                real_images = real_images.to(device)

                # Train Discriminator
                self.netD.zero_grad()
                label = torch.full((real_images.size(0),), self.real_label, dtype=torch.float, device=device)
                output = self.netD(real_images)
                loss_real = criterion(output, label)
                loss_real.backward()

                fake_images = self.netG(skeletons)
                label.fill_(self.fake_label)
                output = self.netD(fake_images.detach())
                loss_fake = criterion(output, label)
                loss_fake.backward()
                optimizerD.step()

                # Train Generator
                self.netG.zero_grad()
                label.fill_(self.real_label)
                output = self.netD(fake_images)
                g_loss = criterion(output, label)
                g_loss.backward()
                optimizerG.step()

            print(f"Epoch [{epoch+1}/{n_epochs}] - D Loss: {(loss_real + loss_fake).item():.4f}, G Loss: {g_loss.item():.4f}")

            
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            if epoch % 10 == 0:
                torch.save(self.netG, self.filename)
            
        print('Finished Training')




    def generate(self, ske):           
        """ generator of image from skeleton """
        
        # ske_t = torch.from_numpy( ske.__array__(reduced=True).flatten() )
        # ske_t = ske_t.to(torch.float32)
        # ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        # normalized_output = self.netG(ske_t)
        # res = self.dataset.tensor2image(normalized_output[0])
        # return res
        # Convert skeleton to a tensor if not already
        ske_t = torch.from_numpy(ske.__array__(reduced=True).flatten()).float().to(next(self.netG.parameters()).device)
        ske_t = ske_t.reshape(1, Skeleton.reduced_dim, 1, 1)  

        # Generate image
        with torch.no_grad():  
            generated_image = self.netG(ske_t).detach().cpu()

        # Convert tensor to image format
        res = self.dataset.tensor2image(generated_image[0])
        return res





if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(200) #5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)