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
from nets.nets_ScenecGAN import UnetUnder, tResUnet2RGB, Unet, ViT
from nets.losses import init_weights, Gradient, GANLoss
from nets.utils import create_loss_meters, update_losses, loss_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())




#Main Model 
class Scene_cGAN(nn.Module):    
    def __init__(self, device=device, lr_Gens= 0.0002 , lr_Disc= 0.0002, 
                 beta1=0.5, beta2=0.999):
        super().__init__()
        
        self.G1 = init_weights(UnetUnder(backbone_name='resnet50'),device)
        self.G2 = init_weights(tResUnet2RGB(),device)
        self.G3 = init_weights(Unet(backbone_name='resnet50'),device)
        self.D1 = init_weights(ViT(),device)
        self.Adv_Loss = GANLoss().to(device)
        self.L1_G1 = nn.L1Loss()
        self.L1_G2 = nn.L1Loss()
        self.cosG3 = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradientG3 = Gradient().to(device)
        self.L1_UW = nn.L1Loss()
        self.opt_Gens = optim.Adam(itertools.chain(self.G1.parameters(),self.G2.parameters(), self.G3.parameters()), lr=lr_Gens, betas=(beta1, beta2))
        self.opt_Discs = optim.Adam(self.D1.parameters(), lr=lr_Disc, betas=(beta1, beta2))
 

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def input_data(self, data):
        self.Gt = data['Gt'].to(device)
        self.Uw = data['Uw'].to(device)
        V = data['V']
        red_max = torch.mean(V[:,0,:,:])
        green_max = torch.mean(V[:,1,:,:])
        blue_max = torch.mean(V[:,2,:,:])
        rgb = [red_max,green_max,blue_max]
        self.veil_vector = torch.tensor(rgb)#.to(device)
        veil_vector = self.veil_vector.unsqueeze(0).expand(BATCH_SIZE,-1).unsqueeze(2).unsqueeze(3).expand(BATCH_SIZE,-1,256,256)
        self.V = veil_vector.to(device)
        self.Depth = data['Depth'].to(device)

        
        
        
    def forward(self):
    
        self.fake_dewater = self.G1(self.Uw)
        self.fv = self.G2(self.Uw)
        fv1 = self.fv.unsqueeze(0).expand(BATCH_SIZE,-1).unsqueeze(2).unsqueeze(3).expand(BATCH_SIZE,-1,256,256)
        self.fake_veil = fv1.to(device)
        self.fake_depth = self.G3(self.Uw)     

        

    
    def backward_D(self):
        #D1 image dewatering G1
        fake_img = self.fake_dewater
        
        #RGB channels G2
        fake_veil = self.fake_veil
             
        #D3 depth map G3
        fake_depth = self.fake_depth
        

        fake_img = fake_img 
        fake_veil = fake_veil 
        fake_depth = fake_depth 


        
        
        depth_matrix = torch.zeros_like(fake_img)
        c=  torch.cat([fake_depth, fake_depth, fake_depth], dim=1)
        depth_matrix[:,0,:,:] = c[:,0,:,:]
        depth_matrix[:,1,:,:] = c[:,1,:,:]
        depth_matrix[:,2,:,:] = c[:,2,:,:]
        #Transmission
        transmission = float(0.001)+torch.exp(-2.2*depth_matrix)
       
        term1 = fake_img * transmission
        t = (1.0-transmission)
        
        term2 = fake_veil * t
        self.UIFM = term1 + term2
         
        #Discriminator fake prediction
        fake_UIFM = torch.cat([self.Uw, self.UIFM], 1) 
        fake_pred_UIFM = self.D1(fake_UIFM.detach())
        self.D1_fake_Loss = self.Adv_Loss(fake_pred_UIFM, False)




        ##Real UIFM
        depth_mat = torch.zeros_like(self.Gt)
        real_d=  torch.cat([self.Depth, self.Depth, self.Depth], dim=1)
        depth_mat[:,0,:,:] = real_d[:,0,:,:]
        depth_mat[:,1,:,:] = real_d[:,1,:,:]
        depth_mat[:,2,:,:] = real_d[:,2,:,:]
        #Transmission
        transmission_r = float(0.001)+torch.exp(-2.2*depth_mat)
        term1_r = self.Gt * transmission_r
        t_r = (1.0-transmission_r)
        term2_r = self.V * t_r
        self.UIFM_real = term1_r + term2_r
         
    

        #Discriminator real prediction
        real_img= torch.cat([self.UIFM_real, self.Uw], 1)
        real_pred_gt = self.D1(real_img)
        self.D1_real_Loss = self.Adv_Loss(real_pred_gt, True)
        self.D1_Loss = (self.D1_fake_Loss + self.D1_real_Loss) * 0.5
        self.D1_Loss.backward()

    def backward_G(self):
        #D1 image dewatering G1
        fake_img = self.fake_dewater
        
        #RGB channels G2
        fake_veil = self.fake_veil
        fv = self.fv     
        #D3 depth map G3
        fake_depth = self.fake_depth
        

        
       #Underwater Image Formation Model
        

        depth_matrix = torch.zeros_like(fake_img)
        c=  torch.cat([fake_depth, fake_depth, fake_depth], dim=1)
        depth_matrix[:,0,:,:] = c[:,0,:,:]
        depth_matrix[:,1,:,:] = c[:,1,:,:]
        depth_matrix[:,2,:,:] = c[:,2,:,:]
       
     
        #Transmission
        transmission = float(0.001)+torch.exp(-1.2*depth_matrix)
        
        term1 = fake_img * transmission
        t = (1.0-transmission)
        term2 = fake_veil * t
        self.UIFM = term1 + term2
      
       

        #Discriminator fake prediction Generator cheating
        fake_UIFM = torch.cat([self.UIFM, self.Uw], 1)       
        fake_pred_UIFM = self.D1(fake_UIFM)
        self.loss_G1_GAN = self.Adv_Loss(fake_pred_UIFM, True) 
        
        depth_grad = self.get_gradientG3(self.Depth)
        fake_grad = self.get_gradientG3(fake_depth)
        ones = torch.ones(self.Depth.size(0), 1, self.Depth.size(2),self.Depth.size(3)).float().to(device).requires_grad_(True)
        depth_grad_fx = depth_grad[:, 0, :, :].contiguous().view_as(self.Depth)
        depth_grad_fy = depth_grad[:, 1, :, :].contiguous().view_as(self.Depth)
        fake_grad_dx = fake_grad[:, 0, :, :].contiguous().view_as(fake_depth)
        fake_grad_dy = fake_grad[:, 1, :, :].contiguous().view_as(fake_depth)

        depth_normal = torch.cat((-depth_grad_fx, -depth_grad_fy, ones), 1)
        output_normal = torch.cat((-fake_grad_dx, -fake_grad_dy, ones), 1)

        loss_depth = torch.log(torch.abs(fake_depth - self.Depth) + 0.5).mean()
        loss_fx = torch.log(torch.abs(fake_grad_dx - depth_grad_fx) + 0.5).mean()
        loss_fy = torch.log(torch.abs(fake_grad_dy - depth_grad_fy) + 0.5).mean()
        loss_normal = torch.abs(1 - self.cosG3(output_normal, depth_normal)).mean()

        loss_tot = loss_depth + loss_normal + (loss_fx + loss_fy)
        
                
        self.G1_Loss = self.L1_G1(fake_img, self.Gt) * 16.0
        self.G2_Loss = self.L1_G2(fv, self.veil_vector) * 15.0
        self.G3_Loss = loss_tot * 11.0
        self.UW_Loss = self.L1_UW(self.UIFM, self.Uw) * 10.0
        self.loss_Gens = self.loss_G1_GAN  + self.UW_Loss + self.G1_Loss + self.G2_Loss + self.G3_Loss
        self.loss_Gens.backward()
        
    
    def optimizer(self):
        self.forward()
        torch.cuda.empty_cache()

        self.D1.train()
        self.set_requires_grad(self.D1, True)
        self.opt_Discs.zero_grad()
        self.backward_D()
        torch.cuda.empty_cache()
        self.opt_Discs.step()
 
        self.G1.train()
        self.G2.train()
        self.G3.train()
        self.set_requires_grad(self.D1, False)
        self.opt_Gens.zero_grad()
        self.backward_G()
        torch.cuda.empty_cache()
        self.opt_Gens.step()



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