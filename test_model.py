"""
    Code for testing Scene-cGAN
"""
import os
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from nets.nets_ScenecGAN import UnetUnder, tResUnet2RGB, Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





G1 = UnetUnder(backbone_name='resnet50').to(device)
G2 = tResUnet2RGB().to(device)
G3=  Unet(backbone_name='resnet50').to(device)

# Download the model weights https://drive.google.com/drive/folders/1gLW6KjeYah3B3sgJYSG3vXED9M758XIv?usp=sharing
#Load state dic
G1.load_state_dict(torch.load('path\net_G1_v10_epoch_20.pth', map_location=device))
G2.load_state_dict(torch.load('path\net_G2_v10_epoch_20.pth', map_location=device))
G3.load_state_dict(torch.load('path\net_G3_v10_epoch_20.pth', map_location=device))


#Input path
input_path = "inputh folder path "

os.chdir(input_path)
img_folder = os.listdir(input_path)

#Output path
output_restored = "Path for restored images"
output_depth = "Path for depth estimation"

#Mode eval
G1.eval()
G3.eval()

#Data transform
transform = transforms.Compose([transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC)])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## testing loop
count = []
img_folder
d=1

for img in img_folder:

    uw_im = transform(Image.open(img))
   
    uw_img = np.array(uw_im).astype("float32")
    uw_input = transforms.ToTensor()(uw_img)
    uw_input = uw_input/255.0
    uw_input = uw_input.unsqueeze_(0)
    uw_input = uw_input.to(device)

    # Generate output dewater image
    fake_dewatered = G1(uw_input)

    # Generate output depth map
    fake_depth = G3(uw_input)
    t = transforms.Grayscale()
    depth_sample = t(fake_depth)
      

    #image_name = (img.split('.')[-2] +'_%d.jpg'%d)
    file_path1 = os.path.join(output_restored, '{}'.format(img))
    file_path2 = os.path.join(output_depth, '{}'.format(img))
    save_image(fake_dewatered, file_path1, normalize=True)
    save_image(depth_sample, file_path2, normalize=True)
    d+=1
    print("Tested: %s" % img)
if (len(count) > 1):
    print("Saved generated images in in %s\n" %(output_restored, output_depth)) 
