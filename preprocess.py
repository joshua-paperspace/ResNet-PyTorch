import torch
from PIL import Image
import torchvision.transforms as transforms

def imgToTensor(image):
    width, height = image.size
    
    # Setting the points for cropped image
    left = 4
    top = height / 5
    right = 154
    bottom = 3 * height / 5
    
    # Cropped image of above dimension
    # (It will not change original image)
    im1 = image.crop((left, top, right, bottom))
    newsize = (224, 224)
    im1 = im1.resize(newsize)

    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # Convert the image to PyTorch tensor
    tensor = transform(im1)
    tensor_shaped = torch.reshape(tensor, [1,3,224,224])

    return tensor_shaped