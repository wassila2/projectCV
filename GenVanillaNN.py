import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from VideoSkeleton import VideoSkeleton
from Skeleton import Skeleton
import os

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    """Transform skeleton data into an image."""
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image


class VideoSkeletonDataset(Dataset):
    """Dataset class for VideoSkeleton."""
    def __init__(self, videoSke, ske_reduced, image_transform=None, target_transform=None):
        self.videoSke = videoSke
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced

    def __len__(self):
        return self.videoSke.skeCount()

    def __getitem__(self, idx):
        ske = self.videoSke.ske[idx]

        # Transform skeleton to input data
        if self.image_transform:
            input_data = self.image_transform(ske)
            input_data = transforms.ToTensor()(input_data).float()
            input_data = (input_data - 0.5) / 0.5  # Normalize to [-1, 1]
        else:
            input_data = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten()).float()

        # Load target image
        image_path = self.videoSke.imagePath(idx)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        target_image = Image.open(image_path)
        if self.target_transform:
            target_image = self.target_transform(target_image)

        return input_data, target_image

    def preprocessSkeleton(self, ske):
        """Preprocess a single skeleton for use as input."""
        ske = torch.from_numpy(ske.__array__(reduced=self.ske_reduced).flatten()).float().view(-1, 1, 1)
        return ske

    def tensor2image(self, tensor):
        """Convert tensor to image format (numpy array)."""
        tensor = (tensor.detach() + 1) / 2  # De-normalize from [-1, 1] to [0, 1]
        tensor = tensor.clamp(0, 1)
        image = tensor.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
        image = (image * 255).astype(np.uint8)  # Convert to uint8 format
        return image


class GenNNSkeToImage(nn.Module):
    """Fully connected model for skeleton-to-image generation."""
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(26, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3 * 64 * 64),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 3, 64, 64)
        return x


class GenNNImageToImage(nn.Module):
    """U-Net model for image-to-image generation."""
    def __init__(self):
        super(GenNNImageToImage, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (3, 64, 64) -> (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (64, 32, 32) -> (128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (128, 16, 16) -> (256, 8, 8)
            nn.ReLU(),
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (256, 8, 8) -> (512, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (512, 4, 4) -> (256, 8, 8)
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (256, 8, 8) -> (128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 16, 16) -> (64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (64, 32, 32) -> (3, 64, 64)
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class GenVanillaNN:
    """Vanilla Generator supporting two types of models."""
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        self.model_type = optSkeOrImage
        self.filename = f"data/GenVanillaNN_model_{optSkeOrImage}.pth"
        self.netG = GenNNSkeToImage() if optSkeOrImage == 1 else GenNNImageToImage()

        image_size = 64
        image_transform = SkeToImageTransform(image_size) if optSkeOrImage == 2 else None
        target_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, image_transform=image_transform, target_transform=target_transform)
        self.dataloader = DataLoader(self.dataset, batch_size=16, shuffle=True)

        if loadFromFile and os.path.exists(self.filename):
            self.netG.load_state_dict(torch.load(self.filename))

    def train(self, n_epochs=20):
        optimizer = optim.Adam(self.netG.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.netG.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for skeletons, images in self.dataloader:
                optimizer.zero_grad()
                inputs = skeletons if self.model_type == 1 else images
                outputs = self.netG(inputs)
                loss = criterion(outputs, images)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {epoch_loss / len(self.dataloader):.4f}")

        torch.save(self.netG.state_dict(), self.filename)
        print(f"Model saved to {self.filename}")

    def generate(self, ske):
        self.netG.eval()  # Set model to evaluation mode
        with torch.no_grad():
            if self.model_type == 1:  # Skeleton input (Model 1)
                input_data = torch.from_numpy(ske.__array__(reduced=True).flatten()).float().unsqueeze(0)
                output = self.netG(input_data)  # Pass skeleton data directly to the model
            elif self.model_type == 2:  # Image input (Model 2)
                # Preprocess the skeleton into an image
                input_image = self.dataset.image_transform(ske)
                input_tensor = transforms.ToTensor()(input_image).float().unsqueeze(0)
                output = self.netG(input_tensor)  # Pass the image to the model
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            tensor_image = output[0]
            return self.dataset.tensor2image(tensor_image)



def display_batch(images):
    """Display a batch of generated images."""
    for img in images:
        cv2.imshow('Generated Image', img)
        if cv2.waitKey(0) & 0xFF == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filename = "data/taichi1.mp4"
    n_epochs = 150
    optSkeOrImage = 2  # 1 for skeleton data, 2 for image with skeleton

    print(f"Checking file existence at: {os.path.abspath(filename)}")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Video file '{filename}' not found. Please check the path.")

    targetVideoSke = VideoSkeleton(filename)

    # Train the model
    gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=optSkeOrImage)
    gen.train(n_epochs=n_epochs)

    # Generate images
    generated_images = []
    for i in range(min(targetVideoSke.skeCount(), 16)):  # Limit for visualization
        image = gen.generate(targetVideoSke.ske[i])
        resized_image = cv2.resize(image, (256, 256))
        generated_images.append(resized_image)

    display_batch(generated_images)
