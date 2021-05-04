import os
import glob
import torch
import numpy as np

val_every = 1 

saved_dir = './saved'
if not os.path.isdir(saved_dir):                                                           
    os.mkdir(saved_dir)
    
def save_model(model, saved_dir, file_name='default.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)


def load_model(model, device, saved_dir, file_name='default.pt'):
    model_path = os.path.join(saved_dir, file_name)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)


def calculate_parameter(model, print_param=False):
    n_param = 0
    n_conv = 0
    for p_idx,(param_name,param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_numpy = param.detach().cpu().numpy() # to numpy array 
            n_param += len(param_numpy.reshape(-1))
            if print_param==True:
                print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
            if "conv" in param_name: n_conv+=1
    print("-"*50+f"\nTotal number of parameters: [{n_param:,d}]\n"+"-"*50)
    print(f"Total number of Conv layer : {n_conv}")