import wandb
import os

import sys, getopt
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import time

from resnet import resnet18, resnet34
from load_data import trainloader, testloader, valloader
from config.config import model_config


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def validate_model(model, valid_dl, loss_func, log_images=False, batch_idx=0):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl, 0):
        # for i, (images, labels) in enumerate(valid_dl), leave=False):
            images, labels = images.to(device), labels.to(device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            # if i==batch_idx and log_images:
                # log_image_table(images, predicted, labels, outputs.softmax(dim=1))
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)


def main(argv):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    wandb.login(key='1305ff3ca47ed8cd6735ba50a3b2f6697ff94916')

    # model=resnet34(3, 10)
    model_dir='model/'
    # model_name='ResNet34'
    # epochs=10

    try:
        opts, args = getopt.getopt(argv,"l:e:",["layers=", "epochs="])
    except:
        print("Could not get any passed arguments.")

    for opt, arg in opts:
        if opt in ('-l', '--layers'):
            if arg == '18':
                model_name = 'ResNet18'
                model = resnet18(3, 10)
            else:
                model_name = 'ResNet34'
                model = resnet34(3, 10)
        if opt in ('-e', '--epochs'):
            epochs = int(arg)
    
    model_config['model'] = model_name
    name = model_name + '-epochs-' + str(epochs)

    with wandb.init(project="resnet-test", config=model_config, name=name):

        print(model_config['epochs'])
        print(model_config['batch_size'])
        print(model_config['lr'])
        print(model_name)

        # if model == 'ResNet18':
        #     model = resnet18(3, 10)
        # else:
        #     model = resnet34(3, 10)

        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.SGD(model.parameters(), lr=model_config['lr'], momentum=0.9)

        model.to(device)

        step = 0
        epoch_durations = []
        # for epoch in range(model_config['epochs']):
        for epoch in range(epochs):   
            start_epoch_time = time.time()
            # print("--- %s seconds ---" % (time.time() - start_time))

            print('epoch:', epoch+1)
            mini_batch_check=50
            running_loss = 0.0
            model.train()

            for i, data in enumerate(trainloader, 0):
                # print(i)

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
                
                if i % mini_batch_check == mini_batch_check-1:    # print every 50 mini-batches
                    step +=1
                    print('inter-epoch:', epoch + ((i+1)/len(trainloader)))
                    wandb.log({"train_loss": running_loss/mini_batch_check, "epoch": epoch + ((i+1)/len(trainloader))}, step=step)

                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / mini_batch_check))
                    
                    running_loss = 0.0
            
            val_loss, accuracy = validate_model(model, valloader, criterion)
                
            # Log validation metrics
            wandb.log({"val_loss": val_loss, "val_accuracy": accuracy}, step=step)
            print(f"Valid Loss: {val_loss:3f}, accuracy: {accuracy:.2f}")
            epoch_duration = time.time() - start_epoch_time
            wandb.log({"epoch_runtime (seconds)": epoch_duration}, step=step)

            epoch_durations.append(epoch_duration)

        avg_epoch_runtime = sum(epoch_durations) / len(epoch_durations)
        wandb.log({"avg epoch runtime (seconds)": avg_epoch_runtime})
        # wandb.finish()

    print('Finished Training')

    PATH = model_dir + name + ".pth"
    torch.save(model.state_dict(), PATH)

    return None

if __name__ == "__main__":
   main(sys.argv[1:])