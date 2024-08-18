"""
Training Scene-cGAN
"""
#Libraries

from tqdm.notebook import tqdm
import torch
from torch import nn, optim
import itertools
import torch.optim as optim

#Local libraries
from data.load_datasets import train_data, BATCH_SIZE
from nets.utils import create_loss_meters, update_losses, loss_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())





#TRAINING

def train_model(model, train_data, epochs=20):
    loss_Gens = []
    D1_Loss = []
    
    for e in range(epochs):
        loss_meter_dict = create_loss_meters()
        i = 0  

        for data in tqdm(train_data):
            model.input_data(data) 
            model.optimizer()
            update_losses(model, loss_meter_dict, count=data['Uw'].size(0)) 
            loss_Gens.append(loss_meter_dict["loss_Gens"].avg)
            D1_Loss.append(loss_meter_dict["D1_Loss"].avg)
            i += 1
            if i % 56 == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_data)}")
                loss_results(loss_meter_dict) 
        torch.save(model.G1.state_dict(), "path_{e+1}.pth")
        torch.save(model.G2.state_dict(), "path_{e+1}.pth")
        torch.save(model.G3.state_dict(), "path_{e+1}.pth")
        torch.save(model.D1.state_dict(), "path_{e+1}.pth")
model = Scene_cGAN()
train_model(model, train_data)
