
from matplotlib import transforms
import torch
import torch.nn as nn
import copy
import numpy as np 
import time
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt

from torchvision.transforms import transforms as torchvision_transforms
from PIL import Image
import json 



def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)

def save_model(model, path):
    print(f'Saving model to {path}...')
    torch.save(model.state_dict(), path)
    print("End of saving !!!")


def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path, map_location={'cuda:0': 'cuda:1'})
    print('------------',path)
    model.load_state_dict(param)
    print("End of loading !!!")



## TO DO ##
def plot_learning_curve(x, y):
    """_summary_
    The function is mainly to show and save the learning curves. 
    input: 
        x: data of x axis 
        y: data of y axis 
    output: None 
    """
    #############
    ### TO DO ### 
    # You can consider the package "matplotlib.pyplot" in this part.
    plt.subplot(2, 2 ,1)
    plt.plot(x, y[0])
    plt.xlabel("epoch")  
    plt.ylabel("acc")  
    plt.title('training acc')

    plt.subplot(2, 2 ,2)
    plt.plot(x, y[1])
    plt.xlabel("epoch")  
    plt.ylabel("loss")  
    plt.title('training loss')

    plt.subplot(2, 2 ,3)
    plt.plot(x, y[2])
    plt.xlabel("epoch")  
    plt.ylabel("acc")  
    plt.title('validation acc')

    plt.subplot(2, 2 ,4)
    plt.plot(x, y[3])
    plt.xlabel("epoch")  
    plt.ylabel("loss")  
    plt.title('validation loss')
   
    plt.tight_layout()
    plt.show()
    pass
    

def train(model, train_loader, val_loader, num_epoch, log_path, save_path, device, criterion, scheduler, optimizer,weakmodel_flag):
    start_train = time.time()

    overall_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_acc = np.zeros(num_epoch ,dtype = np.float32)
    overall_val_loss = np.zeros(num_epoch ,dtype=np.float32)
    overall_val_acc = np.zeros(num_epoch ,dtype = np.float32)

    best_acc = 0

    for i in range(num_epoch):
        print(f'epoch = {i}')
        # epcoch setting
        start_time = time.time()
        train_loss = 0.0 
        corr_num = 0

        # training part
        # start training
        model.train()
        for batch_idx, ( data, label,) in enumerate(tqdm(train_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size

            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 
            # print('batch_idx',batch_idx)

            # calculate the loss between output and ground truth
            loss = criterion(output, label) 
            
            # discard the gradient left from former iteration 
            optimizer.zero_grad()

            # calcualte the gradient from the loss function 
            loss.backward()
            
            # if the gradient is too large, we dont adopt it
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # Update the parameters according to the gradient we calculated
            optimizer.step()

            train_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            # correct if label == predict_label
            corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        scheduler.step()
        
        # averaging training_loss and calculate accuracy
        train_loss = train_loss / len(train_loader.dataset) 
        train_acc = corr_num / len(train_loader.dataset)
                
        # record the training loss/acc
        overall_loss[i], overall_acc[i] = train_loss, train_acc
        
        ## TO DO ##
        # validation part 
        with torch.no_grad():
            model.eval()
            val_loss = 0
            corr_num = 0
            val_acc = 0 
            
            ## TO DO ## 
            # Finish forward part in validation. You can refer to the training part 
            # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

            for batch_idx, ( data, label,) in enumerate(tqdm(val_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
                data = data.to(device)
                label = label.to(device)

            # pass forward function define in the model and get output 
                output = model(data) 

            # calculate the loss between output and ground truth
                loss = criterion(output, label) 
                val_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
                pred = output.argmax(dim=1)
            
            # correct if label == predict_label
                corr_num += (pred.eq(label.view_as(pred)).sum().item())
        
        # averaging training_loss and calculate accuracy
        val_loss = val_loss / len(val_loader.dataset) 
        val_acc = corr_num / len(val_loader.dataset)
                
        # record the training loss/acc
        overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc    

        
        # Display the results
        end_time = time.time()
        elp_time = end_time - start_time
        min = elp_time // 60 
        sec = elp_time % 60
        print('*'*10)
        #print(f'epoch = {i}')
        print('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC '.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
        print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        print('========================\n')

        with open(log_path, 'a') as f :
            f.write(f'epoch = {i}\n', )
            f.write('time = {:.4f} MIN {:.4f} SEC, total time = {:.4f} Min {:.4f} SEC\n'.format(elp_time // 60, elp_time % 60, (end_time-start_train) // 60, (end_time-start_train) % 60))
            f.write(f'training loss : {train_loss}  train acc = {train_acc}\n' )
            f.write(f'val loss : {val_loss}  val acc = {val_acc}\n' )
            f.write('============================\n')

        # save model for every epoch 
        #torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{i}.pt'))
        
        # save the best model if it gain performance on validation set
        if  val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))


    x = range(0,num_epoch)
    overall_acc = overall_acc.tolist()
    overall_loss = overall_loss.tolist()
    overall_val_acc = overall_val_acc.tolist()
    overall_val_loss = overall_val_loss.tolist()
    # Plot Learning Curve
    ## TO DO ##
    # Consider the function plot_learning_curve(x, y) above
    
    y=[overall_acc,overall_loss,overall_val_acc,overall_val_loss]
    if (weakmodel_flag==False):
        plot_learning_curve(x,y)
    pass

def dataCleaning(model, all_set,all_loader, train_set,val_set, device, criterion,threshold=4):
    
    print('----- start Data Cleaning ! -----')
     
    images=all_set.images
    labels=all_set.labels
    cleaned_images=[]
    cleaned_labels=[]
    print(len(images),'len(images)',len(labels),'len(labels)')
    # print(type(images))
    new_train_set=copy.copy(train_set)
    new_val_set=copy.copy(val_set)

    with torch.no_grad():
        model.eval()
        for batch_idx, ( data, label,) in enumerate(tqdm(all_loader)):

        # put the data and label on the device
        # note size of data (B,C,H,W) --> B is the batch size
            data_d = data.to(device)
            label_d = label.to(device)

        # pass forward function define in the model and get output 
            output = model(data_d) 

        # calculate the loss between output and ground truth
            loss = criterion(output, label_d) 
            if  (loss<threshold):
                 cleaned_images.append(images[batch_idx])
                 cleaned_labels.append(labels[batch_idx])
                #  labels.pop(batch_idx-1)
            # else:
                #  print('here!','batch_idx',batch_idx)



    total_len=len(cleaned_images)      
    
    print('----- Data Cleaning done ! -----')
    print('original_data_len : ',len(images),'cleanded_data len : ',total_len)
    new_train_set.images=cleaned_images[int(total_len*0.1):]
    new_train_set.labels=cleaned_labels[int(total_len*0.1):]
    # print(new_train_set.images[1260],new_train_set.images[1261],new_train_set.images[1262])
    new_val_set.images=cleaned_images[:int(total_len*0.1)]
    new_val_set.labels=cleaned_labels[:int(total_len*0.1)]

    return new_train_set, new_val_set
    










