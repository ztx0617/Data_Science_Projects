import argparse
import os

parser = argparse.ArgumentParser(description='Predict the name of plants with probability')

parser.add_argument('image_path', help='The path of image for predicting')
parser.add_argument('checkpoint', help='The path of saved model')
parser.add_argument('--top_k', default=5, type = int, help='Set the top K most likely classes')
parser.add_argument('--category_names', default=os.getcwd()+'/cat_to_name.json', help='The mapping file of categories to real names used')
parser.add_argument('--gpu', action='store_true', help='Whether to use GPU for predicting')


args = parser.parse_args()

print(' The path of image for predicting is: {}\n'.format(args.image_path),
	  'The path of saved model is: {}\n'.format(args.checkpoint),
	  'The top K most likely classes is: {}\n'.format(args.top_k),
	  'The mapping file of categories to real names used is: {}\n'.format(args.category_names),
	  'Use GPU for predicting: {}\n'.format(args.gpu))

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json

# load label mapping
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

 # TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.densenet121(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    if torch.cuda.is_available() and args.gpu:
        model.cuda()
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im = im.resize((224,224))
    np_image = np.array(im)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.array(np_image)- mean)/std
    np_image = np_image.transpose((2,0,1))
    return torch.from_numpy(np_image)

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    if torch.cuda.is_available() and args.gpu:
        image = image.to('cuda',dtype=torch.float)
    else:
        image = image.to('cpu',dtype=torch.float)
    model.eval()
    ps = torch.exp(model.forward(image.unsqueeze_(0)))
    top_p, top_idx = ps.topk(topk, dim=1)
    p_list = list(top_p.cpu().detach().numpy()[0])
    
    idx_to_class = {y:x for x,y in model.class_to_idx.items()}
    class_list = []
    for idx in list(top_idx.cpu().numpy()[0]):
        classes = idx_to_class.get(idx)
        class_list.append(classes)
    return p_list, class_list

# load the model
model = load_checkpoint(args.checkpoint)
# Do prediction
probs, classes = predict(args.image_path, model, topk=args.top_k)

names = []
for clas in classes:
    name = cat_to_name.get(clas)
    names.append(name)

print('The predicted Top {} names and probabilities: {}'.format(args.top_k, dict(zip(names, probs))))