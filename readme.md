
# Posture-Guided Image Synthesis of a Person
- Extract poses from a source video and apply them to a target video to create a new video.
- Use machine learning techniques to generate images of the target person in the extracted poses.
- Implement a simple neural network and a GAN to improve image generation quality.

## Requirements
   - `numpy`
   - `pytorch`
   - `mediapipe` 
   - `OpenCV_python`
## Code Structure

- **VideoReader**: Functions for video playback and image retrieval.
- **Vec3**: Representation of 3D points.
- **Skeleton**: Class that manages 3D skeleton positions(33 points) with an option to reduce to 13 joints in 2D(26 values).
- **VideoSkeleton**: Associates each video frame with a skeleton and stores the images of a video.
- **GenNearest**: A generator that creates an image by finding the nearest skeleton from a video dataset using Euclidean distance, returning the corresponding image.
- **GenVanillaNN**: A neural network-based generator that creates images from skeleton data by transforming skeletons into images using a deep convolutional network. The model is trained with a dataset of skeletons and corresponding images and can generate realistic images from new skeleton poses.
- **GenGAN**: A Generative Adversarial Network (GAN) that generates images from skeleton data. It consists of a generator (GenVanillaNN) that creates images from skeleton poses and a discriminator that distinguishes real from fake images. During training, the generator improves by fooling the discriminator, and the model is optimized using binary cross-entropy loss with Adam. Once trained, it can generate images from new skeleton inputs.
- **DanceDemo**: Main class for the dance demo.

## Approach

### Step 1: Skeleton Extraction

- **Implementation**:
  - Use Mediapipe to extract skeleton data from videos. The code uses `VideoSkeleton` and `VideoReader` to process and extract skeletons from video frames. The video is split into individual frames to process them separately. This matches the code setup where frames are processed to extract skeleton data for training.

### Step 2: Nearest Neighbor Generation

- **Implementation**:
  - This method searches for the most similar skeleton posture in a dataset and retrieves the corresponding image, This would likely involve comparing skeleton features in a feature space and selecting the closest match.

### Step 3: Direct Neural Network

- **Implementation**:
  - The `GenVanillaNN` class implements a simple neural network to generate images from skeletons. The code uses a direct linear  neural  network to convert skeleton data (input as vectors) into images. This aligns with the approach of training a network to directly map skeletons to images.

### Step 4: Neural Network with Stick Figure Image

- **Implementation**:
  
   - The `GenVanillaNN` integrate an intermediate stick figure representation to improve the training process. This is achieved through the conventional neural networks the (`draw_reduced`) method in the `Skeleton` class, which simplifies the skeleton to a stick figure. This stick figure serves as an intermediate visual step, helping the network learn key pose features before generating the final image. By training on this representation, the model can better capture the structure of each pose, making it easier to produce realistic images in the final output.
- 
### Step 5: GAN Implementation

- **Implementation**:
  - This is well implemented in the code with the `GenGAN` class, which adds a discriminator to improve the quality of generated images. The generator (`GenVanillaNN`) creates images from skeletons, while the discriminator differentiates real images from generated ones. This GAN setup allows the model to improve its image generation capability over time.

## Execution
To run this project, follow these steps and ensure to configure the required parameters as described below:
**Run the Nearest Neighbor Generation**
To generate images using the nearest neighbor method:
Modify the DanceDemo.py file to set GEN_TYPE = 1.
**Train the NN models and Running the code**
Training: Use the following command to train the GenVanillaNN model:
`python model_file.py`
The models are trained for 200 epochs with a learning rate (lr) of 0.0001, which yielded the best results during testing.
un the Dance Demo:
After training, run DanceDemo.py after modifiying the  GEN_TYPE to use the specific trained model.
## Performance Comparison:

GenGAN produced the best results among the methods due to its adversarial training (presence of discriminateur), but the generated images remain blurry (flou) and lack sharpness.
GenVanillaNN  produce less realistic images compared to the GAN model.
To visualize results, always update the GEN_TYPE in DanceDemo.py to the corresponding generator (1, 2, 3, or 4) and execute the script.


