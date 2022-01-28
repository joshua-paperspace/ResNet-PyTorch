import sys, getopt
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn

from resnet import resnet18, resnet34
from load_data import trainloader, testloader, valloader

def main(argv):
    model=resnet34(3, 10)
    model_dir='model/'
    model_name='resnet34'
    epochs=1
    try:
        opts, args = getopt.getopt(argv,"l:e:",["layers=", "epochs="])
    except:
        print("Could not get any passed arguments.")
    for opt, arg in opts:
        if opt in ('-l', '--layers'):
            if arg == '18':
                model_name = 'resnet18'
                model = resnet18(3, 10)
            elif arg == '34':
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
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
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
    torch.save(model.state_dict(), PATH)

    return None

if __name__ == "__main__":
   main(sys.argv[1:])