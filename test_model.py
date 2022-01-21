import sys, getopt
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import os

from resnet import resnet18, resnet34
from load_data import testloader

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"m:",["model="])
    except:
        print("Could not get any passed arguments.")
        print("Cannot continue without model.")
        return
    for opt, arg in opts:
        print('in here')
        if opt == '-m':
            print('recognizied m')
            PATH = arg
            print(PATH)
            if PATH[7:15] == 'resnet18':
                model = resnet18(3, 10)
            elif PATH[7:15] == 'resnet34':
                model = resnet34(3, 10)
            else:
                print('why did I hit the else')
                model = resnet18(3, 10)
    
    # model_dir ='models/resnet34'
    # PATH = model_dir + PATH
    model.load_state_dict(torch.load(PATH))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    ## Pretrained Model
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    # myfile = Path('./test_results/' + filename + '.txt')
    # myfile.touch(exist_ok=True)
    # f = open(myfile)

    
    if not os.path.exists('test_results'):
        os.makedirs('test_results')
    filename = PATH[7:-4]
    print("filename: " + filename)
    file = open('test_results/' + filename + '.txt','w+')
    file.write('Network accuracy on test set of 10,000 images: %d%%' % (
        100 * correct / total))
    file.close()

    print('Network accuracy on test set of 10,000 images: %d%%' % (
        100 * correct / total))

    return

    # model=resnet34(3, 10)
    # model_dir='models/'
    # model_name='resnet34'
    # epochs=1
    # try:
    #     opts, args = getopt.getopt(argv,"le:",["layers=", "epochs="])
    # except:
    #     print("Could not get any passed arguments.")
    # for opt, arg in opts:
    #     if opt == '-l':
    #         if arg == '18':
    #             model = resnet18(3, 10)
    #         elif arg == '34':
    #             model = resnet34(3, 10)
    #         else:
    #             model = resnet34(3, 10)
    #     if opt == '-e':
    #         epochs = int(arg)


    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # model.to(device)


    # for epoch in range(epochs):  # loop over the dataset multiple times

    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         # inputs, labels = data
    #         inputs, labels = data[0].to(device), data[1].to(device)

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:    # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                 (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0

    # print('Finished Training')

    # PATH = model_dir + model_name + "-epochs-" + str(epochs) + ".pth"
    # torch.save(model.state_dict(), PATH)

if __name__ == "__main__":
   main(sys.argv[1:])