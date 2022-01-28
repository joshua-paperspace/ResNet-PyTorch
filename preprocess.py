import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

def imgToTensor(image):

    img = cv2.imread('static/uploads/uploaded_image.png', 1)
    img_stretch = cv2.resize(img, (224, 224))
    transform = transforms.ToTensor()
    tensor = transform(img_stretch)
    tensor_shaped = torch.reshape(tensor, [1,3,224,224])

    return tensor_shaped