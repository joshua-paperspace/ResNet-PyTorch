import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2

def imgToTensor(image):


    # img = cv2.imread(image, 1)
    img = cv2.imread('static/uploads/uploaded_image.png', 1)
    print(img.shape)
    # cv2.imshow('Original', img)

    img_stretch = cv2.resize(img, (224, 224))

    # cv2.imshow('Stretched Image', img_stretch)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # width, height = image.size
    
    # Setting the points for cropped image
    # left = 4
    # top = height / 5
    # right = 154
    # bottom = 3 * height / 5
    
    # Cropped image of above dimension
    # (It will not change original image)
    # im1 = image.crop((left, top, right, bottom))
    # newsize = (224, 224)
    # im1 = im1.resize(newsize)

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    # tensor = transform(im1)
    tensor = transform(img_stretch)
    # img_stretch
    tensor_shaped = torch.reshape(tensor, [1,3,224,224])

    return tensor_shaped