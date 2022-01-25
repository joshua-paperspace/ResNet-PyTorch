import sys, getopt
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn

from resnet import resnet18, resnet34
from load_data import trainloader, testloader, valloader

def main(argv):
    model=resnet34(3, 10)
    model_dir='models/'
    model_name='resnet34'
    epochs=1
    try:
        opts, args = getopt.getopt(argv,"l:e:",["layers=", "epochs="])
    except:
        print("Could not get any passed arguments.")
    for opt, arg in opts:
        print('in for loop')
        print('opt: '+ str(opt))
        print('arg:' + str(arg))
        if opt in ('-l', '--layers'):
            print('recognized l')
            if arg == '18':
                print('resnet18')
                model_name = 'resnet18'
                model = resnet18(3, 10)
            elif arg == '34':
                print('resnet34')
                model_name = 'resnet34'
                model = resnet34(3, 10)
            else:
                print('Else resnet34')
                model_name = 'resnet34'
                model = resnet34(3, 10)
        if opt in ('-e', '--epochs'):
            epochs = int(arg)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)


    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = model_dir + model_name + "-epochs-" + str(epochs) + ".pth"

    print('before save')
    torch.save(model.state_dict(), PATH)
    print('after save')

    print(PATH)

    
    # exit(PATH)

    return None

if __name__ == "__main__":
   main(sys.argv[1:])