"""
   Modules for Scene-cGAN paper
   * Paper: 
"""

#Libraries
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class Gaussian_Noise(nn.Module):

    def __init__(self, p_rate: float):
        """
        Multiplicative Gaussian Noise dropout with N(1, p/(1-p))
        It is NOT (1-p)/p like in the paper, because here the
        noise actually increases with p. (It can create the same
        noise as the paper, but with reversed p values)

        Source:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

        :param p: float - determines the the standard deviation of the
        gaussian noise, where sigma = p/(1-p).
        """
        super().__init__()
        assert 0 <= p_rate < 1
        self.weight_mean = torch.ones((0,))
        self.shape = ()
        self.p_rate = p_rate
        self.t_std = self.compute_std()

    def compute_std(self):
        return self.p_rate / (1 - self.p_rate)

    def forward(self, hidden_l):
        if self.training and self.p_rate > 0.:
            if self.weight_mean.shape != hidden_l.shape:
                self.weight_mean = torch.ones_like(input=hidden_l, dtype=hidden_l.dtype, device=hidden_l.device)
            elif self.weight_mean.device != hidden_l.device:
                self.weight_mean = self.weight_mean.to(device=hidden_l.device, dtype=hidden_l.dtype)

            gaussian_noise = torch.normal(self.weight_mean, self.t_std)
            hidden_l = hidden_l.mul(gaussian_noise)
        return hidden_l



#Batch Normalization 
class BatchNorm(nn.Module):
    def __init__(self, in_c):
        super().__init__()       
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.LeakyReLU(0.2, True)
        
    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.relu1(x)
        return x
    
# Residual Unit
class Residual_Unit(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.batch2 = BatchNorm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batch2(x)
        x = self.conv2(x)
        s = self.s(inputs)
        skip = x + s
        return skip

class Decoder1(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.uprelu = nn.ReLU(True)
        self.res = Residual_Unit(in_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = self.uprelu(x)
        x = torch.cat([x, skip], axis=1)
        x = self.res(x)
        return x


class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.batchN1 = BatchNorm(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.batchN2 = BatchNorm(out_c)

        

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batchN1(x)
        x = self.conv2(x)
        x = self.batchN2(x)

        return x

class tResUnet2RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_conv = Conv_Block(1,1)
        self.conv1 = Conv_Block(1,64)
        self.conv2 = nn.Conv2d(1, 64, kernel_size=1, padding=0)
        self.droput = Gaussian_Noise(p_rate=0.5)
        self.res2 = Residual_Unit(64, 128, stride=2)
        self.res3 = Residual_Unit(128, 256, stride=2)
        self.res4 = Residual_Unit(256, 3, stride=2)
        self.output = nn.Linear(32, 1)  
        self.relu = nn.ReLU()


    def forward(self, inputs):
        #RGB channels
        red_channel = inputs[:,0,:,:].unsqueeze(1) 
        green_channel = inputs[:,1,:,:].unsqueeze(1)
        blue_channel = inputs[:,2,:,:].unsqueeze(1)
        red_output = self.c_conv(red_channel)
        green_output = self.c_conv(green_channel)
        blue_output = self.c_conv(blue_channel)

        x = self.conv1(red_output)
        s = self.conv2(red_output)
        skip = x + s
        skip1 = self.droput(skip)
        skip2 = self.res2(skip1)
        skip3 = self.res3(skip2)
        b = self.res4(skip3)
        output = self.output(b)
        output_r = self.relu(output)
        output_r = output_r.view(output_r.size(1), -1)
        output_r = torch.max(output_r[0])

        x1 = self.conv1(green_output)
        s1 = self.conv2(green_output)
        skipg = x1 + s1
        skip1g = self.droput(skipg)
        skip2g = self.res2(skip1g)
        skip3g = self.res3(skip2g)
        bg = self.res4(skip3g)
        outputg = self.output(bg)
        output_g = self.relu(outputg)
        output_g = output_g.view(output_g.size(1), -1)
        output_g = torch.max(output_g[0])

        x2 = self.conv1(blue_output)
        s2 = self.conv2(blue_output)
        skipb = x2 + s2
        skip1b = self.droput(skipb)
        skip2b = self.res2(skip1b)
        skip3b = self.res3(skip2b)
        bb = self.res4(skip3b)
        outputb = self.output(bb)
        output_b = self.relu(outputb)
        output_b = output_b.view(output_b.size(1), -1)
        output_b = torch.max(output_b[0])
        out = [output_r, output_g, output_b]
        out = torch.tensor(out)

        return out

#Backbone

def get_backbone(name):

    """ Loading backbone, defining names for skip-connections and encoder output. 
        Source: https://github.com/mkisantal/backboned-unet/tree/master
    """

    # TODO: More backbones
    
    weights = ResNet50_Weights.DEFAULT
    # loading backbone model
    if name == 'resnet50':
        backbone = resnet50(weights=weights)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output


class UpsampleBlock(nn.Module):


    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, parametric=False):
        super(UpsampleBlock, self).__init__()

        self.parametric = parametric

        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        if parametric:
            # versions: kernel=4 padding=1, kernel=2 padding=0
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        else:
            self.up = None
            ch_in = ch_in + skip_in
            self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                                   stride=1, padding=1, bias=(not use_bn))
            
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        self.droput = Gaussian_Noise(p_rate=0.5)
        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out if not parametric else ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None
      
    def forward(self, x, skip_connection=None):

        x = self.up(x) if self.parametric else F.interpolate(x, size=None, scale_factor=2, mode='bilinear',
                                                             align_corners=None)
        if self.parametric:
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)
            x = self.droput(x)

        if not self.parametric:
            x = self.conv1(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x


class Unet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name='resnet50',
                 encoder_freeze=False,
                 classes=1,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        super(Unet, self).__init__()

        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):

        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(*input)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 256, 256)
        self.backbone_name == 'unet_encoder'
        channels = [0]  

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param


class UnetUnder(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name='resnet50',
                 encoder_freeze=False,
                 classes=3,
                 decoder_filters=(256, 128, 64, 32, 16),
                 parametric_upsampling=True,
                 shortcut_features='default',
                 decoder_use_batchnorm=True):
        super(UnetUnder, self).__init__()

        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name)
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)
        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      parametric=parametric_upsampling,
                                                      use_bn=decoder_use_batchnorm))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

        if encoder_freeze:
            self.freeze_encoder()

        self.replaced_conv1 = False  # for accommodating  inputs with different number of channels later

    def freeze_encoder(self):

        """ Freezing encoder parameters, the newly initialized decoder parameters are remaining trainable. """

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):

        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(*input)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):

        """ Forward propagation in backbone encoder network.  """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 256, 256)
        self.backbone_name == 'unet_encoder'
        channels = [0]  

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param


# Discriminator ViT

class PatchEmbedding(nn.Module):
    """
    Args:
        input_channels (int): input channels.
        patch_size (int): patches size.
        embedding_dim (int): embedding size.
    """ 
    def __init__(self, input_channels:int=6, patch_size:int=16, embedding_dim:int=768):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.patches = nn.Conv2d(input_channels, embedding_dim, kernel_size=patch_size,
                                 stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
    def forward(self, inputs):
        patches = self.patches(inputs)
        flattened = self.flatten(patches) 
        return flattened.permute(0, 2, 1) # ouput shape
    


class ViT(nn.Module): 
  def __init__(self,
               img_size=256, 
               channels=6,
               patch_size=16,
               embedding_dim=768, 
               dropout=0.1, 
               mlp_size=3072, 
               transformer_layers=12, 
               num_heads=12,
               num_classes=2): 
    super().__init__()


    self.patch_embedding = PatchEmbedding(input_channels=channels,
                                          patch_size=patch_size,
                                          embedding_dim=embedding_dim)

    self.token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                    requires_grad=True)

    num_patches = (img_size * img_size) // patch_size**2
    self.positional_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim))

    self.embedding_dropout = nn.Dropout(p=dropout)
    

    self.transformer_encoder = nn.TransformerEncoder(encoder_layer=nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                                              nhead=num_heads,
                                                                                              dim_feedforward=mlp_size,
                                                                                              activation="gelu",
                                                                                              batch_first=True,
                                                                                              norm_first=True), # Create a single Transformer Encoder Layer
                                                     num_layers=transformer_layers) 

    self.mlp_head = nn.Sequential(
        nn.LayerNorm(normalized_shape=embedding_dim),
        nn.Linear(in_features=embedding_dim, out_features=num_classes)
    )

  def forward(self, inputs):
    batch_size = inputs.shape[0]
    x = self.patch_embedding(inputs)

    token = self.token.expand(batch_size, -1, -1) 


    x = torch.cat((token, x), dim=1)
 
    x = self.positional_embedding + x
  
    x = self.embedding_dropout(x)

    x = self.transformer_encoder(x)

    x = self.mlp_head(x[:, 0])

    return x