
import torch
import os


from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
from torchsummary import summary
from myModels import  myLeNet, myResnet
from myDatasets import  get_cifar10_train_val_set
from tool import train, fixed_seed,load_parameters,dataCleaning
import torchvision.models as models

# Modify config if you are conducting different models
from cfg import resnet18_cfg as cfg
# from cfg import LeNet_cfg as cfg
# from cfg import Resnet_cfg as cfg


def train_interface():
    
    """ input argumnet """
    data_root = cfg['data_root']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs( os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs( os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)


    with open(log_path, 'w'):
        pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    milestones = cfg['milestones']
    
    
    ## Modify here if you want to change your model ##
    # model = myResnet(num_out=num_out)
    model = models.resnet18(pretrained=True)
    # print model's architecture
    print(model)
    # summary(model,(3,32,32))
    
    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    train_set, val_set,all_set =  get_cifar10_train_val_set(root=data_root, ratio=split_ratio)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    all_loader = DataLoader(all_set,batch_size=1  ,shuffle=True)

    # define your loss function and optimizer to unpdate the model's parameters.
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    #原gamma=0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)
    
    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ### 
    # Complete the function train
    # # Check tool.py

    print('--------- weak_model train! -------------')

    # train(model=model, train_loader=train_loader, val_loader=val_loader, 
    #       num_epoch=num_epoch, log_path=log_path, save_path=save_path,
    #       device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler,weakmodel_flag=True)

    # load weak model 
    load_parameters(model=model, path='./save_dir/resnet18/best_model.pt')
    model.to(device)

    # dataCleaning
    new_train_set, new_val_set = dataCleaning(model=model,all_set= all_set,all_loader=all_loader, train_set=train_set, val_set=val_set, device=device, criterion=criterion,threshold=2)
    
    print(new_train_set,len(new_train_set),type(new_train_set),'new_train_set , len , type')
    print(new_val_set,len(new_val_set),type(new_val_set),'new_val_set , len , type')


    new_train_loader = DataLoader(new_train_set, batch_size=batch_size, shuffle=True)
    new_val_loader = DataLoader(new_val_set, batch_size=batch_size, shuffle=False)      

    # model
    print('--------- real_model train! -------------')

    model=models.resnet18(pretrained=True).to(device)
    # real_model = myResnet(num_out=num_out)
    # print(real_model)

    # define your loss function and optimizer to unpdate the model's parameters.
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,weight_decay=1e-6, nesterov=True)
    #原gamma=0.1
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=milestones, gamma=0.1)

    train(model=model, train_loader=new_train_loader, val_loader=new_val_loader, 
          num_epoch=num_epoch, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler,weakmodel_flag=False)

    
if __name__ == '__main__':
    train_interface()




    