

## You could add some configs to perform other training experiments...

LeNet_cfg = {
    'model_type': 'LeNet',
    'data_root' : './p2_data/annotations/train_annos.json',
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}

Resnet_cfg = {
    'model_type': 'Resnet',
    'data_root' : './p2_data/annotations/train_annos.json',
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 150,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 25],
    'num_out': 10,
    'num_epoch': 30,
    
}


resnet18_cfg = {
    'model_type': 'resnet18',
    'data_root' : './p2_data/annotations/train_annos.json',
    # ratio of training images and validation images 
    'split_ratio': 0.9,
    # set a random seed to get a fixed initialization 
    'seed': 687,
    
    # training hyperparameters
    'batch_size': 16,
    'lr':0.01,
    'milestones': [15, 20],
    'num_out': 10,
    'num_epoch': 25,
    
}
