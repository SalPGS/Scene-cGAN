"""
Dataloader Scene-cGAN

"""
#Libraries
import os
from glob import glob
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn


#Data sets
groundtruth = "path Ground_Truth" 
path_gt = glob.glob(groundtruth + "\\*.jpg") 
underwater = "path Underwater" 
path_underwater = glob.glob(underwater + "\\*.jpg") 
airlight = "path Veiling_Light" 
path_airlight = glob.glob(airlight + "\\*.jpg") 
depth_map = "path Depth"
path_depth = glob.glob(depth_map + "\\*.jpg")


# Seed
seed = 75
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = False
cudnn.deterministic = True

#1st dataset is groundtruth dataset which is the gorund truth
np.random.seed(seed)
path_subset_gt = np.random.choice(path_gt, 57960, replace=False) #57960
rand_idxs = np.random.permutation(57960) #57960
train_idxs = rand_idxs[:46368]#46368 
test_idxs = rand_idxs[46368:]
train_path_gt = path_subset_gt[train_idxs]
test_path_gt = path_subset_gt[test_idxs]
print(len(train_path_gt), len(test_path_gt))

#2nd dataset is the underwater dataset 
np.random.seed(seed)
path_subset_underwater = np.random.choice(path_underwater, 57960, replace=False)
rand_underwater_idxs = np.random.permutation(57960)
train_underwater_idxs = rand_underwater_idxs[:46368] 
test_underwater_idxs = rand_underwater_idxs[46368:] 
train_path_underwater = path_subset_underwater[train_underwater_idxs]
test_paths_underwater = path_subset_underwater[test_underwater_idxs]
print(len(train_path_underwater), len(test_paths_underwater))

#3rd dataset is the airlight
np.random.seed(seed)
path_subset_airlight = np.random.choice(path_airlight, 57960, replace=False) 
rand_airlight_idxs = np.random.permutation(57960)
train_airlight_idxs = rand_airlight_idxs[:46368] 
test_airlight_idxs = rand_airlight_idxs[46368:] 
train_path_airlight = path_subset_airlight[train_airlight_idxs]
test_paths_airlight = path_subset_airlight[test_airlight_idxs]
print(len(train_path_airlight), len(test_paths_airlight))

#4th dataset is the depth
np.random.seed(seed)
path_subset_depth = np.random.choice(path_depth, 57960, replace=False) 
rand_depth_idxs = np.random.permutation(57960)
train_depth_idxs = rand_depth_idxs[:46368] 
test_depth_idxs = rand_depth_idxs[46368:]
train_path_depth = path_subset_depth[train_depth_idxs]
test_paths_depth = path_subset_depth[test_depth_idxs]
print(len(train_path_depth), len(test_paths_depth))


img_size = 256
BATCH_SIZE = 6

class ScenecGAN_Dataset(Dataset):

    def __init__(self, path_under, path_gt, path_airlight, path_depth):

        """
        Args:
            path_under: path to underwater images folder.
            path_gt: path to in-air images folder.
            path_airlight: veiling light folder.
            path_depth: veiling light folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transforms = transforms.Compose([
                transforms.Resize((img_size, img_size), transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),


            ])

        self.size = img_size
        self.path_under = path_under
        self.path_gt = path_gt
        self.path_airlight = path_airlight
        self.path_depth = path_depth



    def __getitem__(self, idx):
   

        img_w = Image.open(self.path_under[idx])
        img_grt = Image.open(self.path_gt[idx])
        img_a = Image.open(self.path_airlight[idx])
        img_d = Image.open(self.path_depth[idx])
        img_background = Background_light(img_w)
        

        state = torch.get_rng_state()
        img_water = self.transforms(img_w)
        torch.set_rng_state(state)
        img_gt = self.transforms(img_grt)
        torch.set_rng_state(state)
        img_depth = self.transforms(img_d)
        torch.set_rng_state(state)
        img_background = self.transforms(img_background)

        img_with_water = np.array(img_water).astype("float32")
        img_with_water = transforms.ToTensor()(img_with_water)

        img_ground_truth = np.array(img_gt).astype("float32")
        img_ground_truth = transforms.ToTensor()(img_ground_truth)
        
        img_back = np.array(img_background).astype("float32")
        img_back = transforms.ToTensor()(img_back)

        img_airl = self.transforms(img_a)
        img_airlight = np.array(img_airl).astype("float32")
        img_airlight = transforms.ToTensor()(img_airlight)


        img_depth_map = np.array(img_depth).astype("float32")
        img_depth_map = transforms.ToTensor()(img_depth_map)
         
        
        Uw= img_with_water /255.0
        Gt = img_ground_truth /255.0
        V = img_airlight /255.0
        Depth = img_depth_map /255.0
        Back = img_back/255.0






        return {'Uw': Uw, 'Gt': Gt,  'V': V, 'Depth': Depth, 'Back': Back}



    def __len__(self):
        #len(self.path_under)
        return len(self.path_depth)



def load_data(batch_size=BATCH_SIZE, n_workers=0, pin_memory=True, shuffle=True,**kwargs):
    dataset_images = ScenecGAN_Dataset(**kwargs)
    dataloader= DataLoader(dataset_images, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory, shuffle=shuffle, drop_last=True)
    return dataloader


train_data = load_data( path_under= train_path_underwater, path_gt=train_path_gt, path_airlight=train_path_airlight, path_depth=train_path_depth)
