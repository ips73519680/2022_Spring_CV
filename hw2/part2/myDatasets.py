

import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np 

from torchvision.transforms import transforms
from PIL import Image
import json 


def get_cifar10_train_val_set(root, ratio=0.9, cv=0):
    
    # get all the images path and the corresponding labels
    with open(root, 'r') as f:
        data = json.load(f)
    images, labels = data['images'], data['categories']

    
    info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
    N = info.shape[0]

    # apply shuffle to generate random results 
    np.random.shuffle(info)
    x = int(N*ratio) 
    
    all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


    train_image = all_images[:x]
    # train_image[0]= 513008.jpg
    val_image = all_images[x:]

    train_label = all_labels[:x] 
    val_label = all_labels[x:]
    

    
    ## TO DO ## 
    # Define your own transform here 
    # It can strongly help you to perform data augmentation and gain performance
    # ref: https://pytorch.org/vision/stable/transforms.html
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
                ## TO DO ##
                # You can add some transforms here
                transforms.RandomHorizontalFlip(p=0.5),        
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomInvert(p=0.5),
        
                # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
                transforms.ToTensor(),
                # experimental normalization for image classification 
                transforms.Normalize(means, stds),
            ])
  
    # normally, we dont apply transform to test_set or val_set
    val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ])

 
  
    ## TO DO ##
    # Complete class cifiar10_dataset
    train_set, val_set,all_set = cifar10_dataset(images=train_image, labels=train_label,transform=train_transform), \
                        cifar10_dataset(images=val_image, labels=val_label,transform=val_transform),\
                        cifar10_dataset(images=all_images, labels=all_labels,transform=val_transform)
    return train_set, val_set,all_set



## TO DO ##
# Define your own cifar_10 dataset
class cifar10_dataset(Dataset):
    def __init__(self,images , labels=None , transform=None, prefix = './p2_data/train'):
        
        # It loads all the images' file name and correspoding labels here
        self.images = images 
        self.labels = labels 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = prefix
        
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        image_file_name=self.images[idx]
        path = os.path.join(self.prefix, image_file_name)
        image = Image.open(path).convert('RGB')
        label=self.labels[idx]
        # print('image',image)
        # image <PIL.Image.Image image mode=RGB size=32x32 at 0x27E33F87E20>
        if self.transform:
            image = self.transform(image)

        # You shall return image, label with type "long tensor" if it's training set        
        return image,label


        