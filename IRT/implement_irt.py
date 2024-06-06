'''
Code for implementing 1-PL Item Response Theory (student ability and item difficulty) in Python
'''
import os
import random 
import numpy as np
import json 
import torch 
from torch.utils.data import TensorDataset, DataLoader, random_split

# setting the seed
def set_seed(seed_val=37):
    '''
    set random seed for reproducibility
    '''
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def read_dataset(num_items):
    data_path = 'IRT_dataset/code67/IRT_dataset.json'
    with open(data_path) as f:
        data = json.load(f)
    student_ids = []
    outputs = []
    for student_id, student_data in data.items():
        student_ids.append(student_id)
        outputs.append(list(student_data.values())[:num_items])
    print(*student_ids)
    return student_ids, outputs

class IRTModel(torch.nn.Module):
    def __init__(self, num_students, num_items, load_params=False):
        super(IRTModel, self).__init__()
        self.num_students = num_students
        self.num_items = num_items
        self.student_ability = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (self.num_students,)))
        if not load_params:
            self.item_difficulty = torch.nn.parameter.Parameter(torch.normal(0.0, 0.1, (self.num_items,)))
        else:
            # load the saved parameters and freeze them
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            if device == torch.device('cpu'):
                self.item_difficulty = torch.load('IRT/IRT_parameters/item_difficulty.pt', map_location=torch.device('cpu')).requires_grad_(False)
            else:
                self.item_difficulty = torch.load('IRT/IRT_parameters/item_difficulty.pt').requires_grad_(False)
    
    def forward(self, student_ids, item_ids):
        '''
        student_ids and item_ids are not of the same size
        '''
        student_ability = self.student_ability[student_ids]
        # broadcase student_ability to the size of item_difficulty
        student_ability = student_ability.unsqueeze(1).expand(-1, len(item_ids))
        item_difficulty = self.item_difficulty[item_ids]
        predictions = student_ability - item_difficulty
        return predictions

def play_with_model(model):

    # play with the model 
    student_ids = torch.tensor([0, 1])
    item_ids = torch.tensor([0, 1, 2])

    predictions = model(student_ids, item_ids)
    print('Sample Predictions Shape: ', predictions.shape)

def get_dataloader(batch_size, student_ids, outputs):
    output = torch.tensor(outputs, dtype=torch.float32)
    student_ids = torch.tensor(student_ids, dtype=torch.int64)
    data = TensorDataset(student_ids, output)
    return DataLoader(data, batch_size=batch_size, shuffle=True)

def get_model_info(num_students, num_questions, load_params=False, verbose=True):
    '''
    Return IRT model and the optimizers
    '''
    # select device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if verbose:
        print('Using device:', device)
    # read model
    model = IRTModel(num_students, num_questions, load_params).to(device)
    # loss fucntion
    loss_fn = torch.nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-3)
    # number of training epochs
    num_epochs = 1000
    return model, loss_fn, optimizer, num_epochs, device
    
def train_IRT(item_ids_lst, model, loss_fn, optimizer, num_epochs, device, train_dataloader, verbose=True):
    '''
    Train the model
    '''
    item_ids = torch.tensor(item_ids_lst, dtype=torch.int64).to(device)
    for epoch in range(num_epochs):
        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            # Iterate over data.
            for student_ids, output in train_dataloader:
                student_ids = student_ids.to(device)
                output = output.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(student_ids, item_ids)
                    loss = loss_fn(predictions, output)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            if verbose:
                print('Loss: ', loss.item())            
    return model