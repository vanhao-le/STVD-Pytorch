import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    training_log = {'epoch': [], 'training_loss': [], 'val_loss': [], 'best_loss': 1.0}
    best_model_wts = copy.deepcopy(model.state_dict())

    best_loss = 9999

    for epoch in range(num_epochs):
        training_log['epoch'].append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode
            
            running_loss = 0.0
            # Iterate over data
            for (data, target) in dataloaders[phase]:

                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                               
                data = tuple(d.to(device) for d in data)
                if target is not None:
                    target = target.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                is_train = True if(phase == 'train') else False
                with torch.set_grad_enabled(is_train):  
                    # *==> unpacking tuple                   
                    outputs = model(*data)
                    
                    if type(outputs) not in (tuple, list):
                        outputs = (outputs,)

                    # print(*outputs)
                    # loss_inputs = outputs
                    # if target is not None:
                    #     target = (target,)
                    #     loss_inputs += target

                    loss_outputs = criterion(*outputs)

                    loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs                    
                                       
                    # backward + optimize only if in training phase
                    if(phase == 'train'):
                        loss.backward()
                        optimizer.step()
                               
              
                running_loss += loss.item()
               
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
            
            
            epoch_loss = running_loss / dataset_sizes[phase]            

            print('{} loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'train':
                training_log['training_loss'].append(epoch_loss)          
            if phase == 'val':
                training_log['val_loss'].append(epoch_loss)
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60 ))

    print('Best val loss: {:4f}'.format(best_loss))
    training_log['best_loss'] = best_loss.to('cpu').numpy()
    training_log['training_loss'] = np.array(training_log['training_loss'])
    training_log['val_loss'] = np.array(training_log['val_loss'])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log