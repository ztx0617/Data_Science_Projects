import argparse
import os

parser = argparse.ArgumentParser(description='Train the Neural Network model')

parser.add_argument('data_directory', help='Set the directory of pictures')
parser.add_argument('--save_dir', default=os.getcwd(), help='Set the directory of saving checkpoints')
parser.add_argument('--arch', default='densenet121', choices=['densenet121', 'vgg13'], 
	help='Choose the architecture of pre-trained model. Please choose from densenet121 and vgg13')
parser.add_argument('--learning_rate', default=0.003, type = float,  help='Set the learning rate')
parser.add_argument('--hidden_units', default=500, type = int, help='Set the number of hidden units')
parser.add_argument('--epochs', default=3, type = int, help='Set the number of epochs')
parser.add_argument('--gpu', action='store_true', help='Whether to use GPU for training')

args = parser.parse_args()

print(' The directory of pictures is: {}\n'.format(args.data_directory),
	  'The directory of saving checkpoints is: {}\n'.format(args.save_dir),
	  'The architecture of pre-trained model is: {}\n'.format(args.arch),
	  'The learning rate is: {}\n'.format(args.learning_rate),
	  'The number of hidden units is: {}\n'.format(args.hidden_units),
	  'The number of epochs is: {}\n'.format(args.epochs),
	  'Use GPU for training: {}\n'.format(args.gpu))


import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)

print('Data transformed and loaded')

# TODO: Build and train your network
if args.gpu and torch.cuda.is_available():
	device = torch.device('cuda')
	print('GPU will be used')
else:
	device = torch.device('cpu')
	print('CPU will be used')

if args.arch == 'densenet121':
	model = models.densenet121(pretrained=True)
	print('The densenet121 will be used')
else:
	model = models.vgg13(pretrained=True)
	print('The vgg13 will be used')


for param in model.parameters():
    param.requires_grad = False


model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, args.hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(args.hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

model.to(device)

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    validation_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {validation_loss/len(validationloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()

print('Training finished')
# Save the checkpoint
model.class_to_idx = train_data.class_to_idx
checkpoint = {'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}
torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
print('Model saved to: ' + args.save_dir + '/checkpoint.pth')