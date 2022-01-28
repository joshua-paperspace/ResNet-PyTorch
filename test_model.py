import sys, getopt
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os
import json

from resnet import resnet18, resnet34
from load_data import testloader

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"m:i:",["model=", "id="])
    except:
        print("Could not get any passed arguments.")
        print("Cannot continue without model.")
        return
    for opt, arg in opts:
        print('in here')
        if opt in ('-m', '--model'):
            PATH = arg
            if PATH[14:22] == 'resnet18':
                model = resnet18(3, 10)
            elif PATH[14:22] == 'resnet34':
                model = resnet34(3, 10)
            else:
                model = resnet18(3, 10)
        elif opt in ('-i', '--id'):
            model_id = arg

    model.load_state_dict(torch.load(PATH))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    
    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if not os.path.exists('test-results'):
        os.makedirs('test-results')
    
    filename = PATH[14:-4]

    test_results_json = {
        'model_name': filename
        ,'accuracy': 100 * correct / total
        ,'model_id': model_id
    }

    with open('test-results/' + filename + '.json', 'w') as outfile:
        json.dump(test_results_json, outfile)
    
    print('Network accuracy on test set of 10,000 images: %d%%' % (
        100 * correct / total))

    return None

if __name__ == "__main__":
   main(sys.argv[1:])