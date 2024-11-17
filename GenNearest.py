
import numpy as np
import cv2
import os
import pickle
import sys
import math
from PIL import Image

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it select the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt


    def euclidean_distance(self, skeleton1, skeleton2):
        """
        Compute the Euclidean distance between two skeletons.
        Assumes skeleton1 and skeleton2 are arrays of keypoints.
        """
        return np.linalg.norm(np.array(skeleton1) - np.array(skeleton2))

    def generate(self, ske):
        """ Generator of image from skeleton using nearest neighbor search """
        
        # Initialize variables to store the best match
        nearest_image = None
        min_distance = float('inf')

        # Iterate over all target skeletons and find the closest one
        for i in range(self.videoSkeletonTarget.skeCount()):  
            target_skeleton = self.videoSkeletonTarget.ske[i]  
            target_image = self.videoSkeletonTarget.imagePath(i)  
            
            # Assuming you have a method to load the image
            loaded_image = Image.open(target_image)
            loaded_image = np.array(loaded_image)  
            
            distance = self.euclidean_distance(ske, target_skeleton)
            
            if distance < min_distance:
                min_distance = distance
                nearest_image = loaded_image  

        # If a match is found, return the nearest image
        if nearest_image is not None:
            return nearest_image

        # If no match is found, return a default empty image (as a fallback)
        empty = np.ones((64, 64, 3), dtype=np.uint8) * 255  
        return empty
