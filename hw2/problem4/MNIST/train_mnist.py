import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, Fully
from data import get_dataloader

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    folder, model_type = sys.argv[1], sys.argv[2]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'fully':
        model = Fully()

    # Set the type of gradient optimizer and the model it update 
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # Set history
    train_loss_history, train_acc_history = [], []
    valid_loss_history, valid_acc_history = [], []

    # Run any number of epochs you want
    ep = 10
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############
        
        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0
        
        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader,1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)
            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch           
                print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))

        train_loss_history.append(total_loss / len(train_loader))
        train_acc_history.append(correct_cnt / total_cnt)

        ################
        ## Validation ##
        ################
        model.eval()
        val_loss = 0
        val_cnt = 0
        val_correct = 0
        with torch.no_grad():
            for x, label in val_loader:
                if use_cuda:
                    x, label = x.cuda(), label.cuda()
                out = model(x)
                loss = criterion(out, label)
                val_loss += loss.item()
                _, pred_label = torch.max(out, 1)
                val_cnt += x.size(0)
                val_correct += (pred_label == label).sum().item()

            print ('validation loss: {:.6f}, acc: {:.3f}'.format(
                    val_loss/len(val_loader), val_correct/val_cnt))

            valid_loss_history.append(val_loss / len(val_loader))
            valid_acc_history.append(val_correct / val_cnt)

        model.train()

    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Plot Learning Curve
    if model_type == 'conv':
        net_name = 'Convolutional Network'
    elif model_type == 'fully':
        net_name = 'Fully Connected Network'
    # Plot Loss Curve
    plt.figure()
    plt.plot(range(ep), train_loss_history, c='blue', label='train')
    plt.plot(range(ep), valid_loss_history, c='red', label='valid')
    plt.title('Loss Curve of ' + net_name)
    plt.xlabel('epoch')
    plt.ylabel('cross entropy')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

    # Plot Accuracy Curve
    plt.figure()
    plt.plot(range(ep), train_acc_history, c='blue', label='train')
    plt.plot(range(ep), valid_acc_history, c='red', label='valid')
    plt.title('Accuracy Curve of '+ net_name)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('acc.png')
    plt.close()
