import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from pdb import set_trace as st
import functools

###############################################################################
# Functions
###############################################################################

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or  classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_clamp(m, c=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.clamp_(-c, c)
        if m.bias is not None:
            m.bias.data.clamp_(-c,c)
    elif classname.find('BatchNorm2d') != -1:
        if m.weight is not None:
            m.weight.data.clamp_(-c, + c)
            m.bias.data.clamp_(-c,c)

def define_G(nz, nc, nf, imageSize, n_layers, n_residual, tensor, upsampling=True, norm=nn.BatchNorm2d, non_linearity=nn.ReLU(), up_to=2, noise=0, dropout=0, up=True):
    stacks = np.log2(imageSize)
    assert int(stacks) == stacks, 'Image not power of 2 : {}' % (imageSize)
    stacks = int(stacks)

    stack_dim = nf * 2 ** (stacks - 2) if up == True else nf
    current_image_size = 1 if up else imageSize
    layers = []
    fmc = functools.partial(FeatureMapChanger, upsampling=upsampling, up=up, norm_layer=norm)
    for stack in range(1, stacks + 1):
        if stack >= up_to:
            non_linearity = nn.ReLU

        if up:
            dim_in_stack = nz if stack == 1 else stack_dim
            dim_out_stack = stack_dim if stack == 1 else stack_dim // 2
            if stack == stacks:
                dim_out_stack = nc
        else:
            dim_in_stack = nc if stack == 1 else stack_dim
            dim_out_stack = stack_dim if stack == 1 else stack_dim * 2
            if stack == stacks:
                dim_out_stack = nz

        layers += fmc(dim_in_stack, dim_out_stack, non_linearity=non_linearity, current_image_size=current_image_size)
        current_image_size = current_image_size * 2 if up else current_image_size // 2
        if noise > 0:
            layers += [Noiser(noise, tensor)]
        layers = layers + [nn.Dropout2d(p=dropout)] if dropout > 0 else layers
        layers += [ResnetBlock(dim_out_stack, norm) for residual in range(n_residual)]
        for layer in range(n_layers):
            layers += [CoBlock(dim_out_stack, dim_out_stack, norm, non_linearity)]
            layers += [ResnetBlock(dim_out_stack, norm) for residual in range(n_residual)]

        if stack > 1:
            stack_dim = stack_dim // 2 if up else stack_dim * 2

    if up:
        layers += [nn.Tanh()]
    else:
        layers += [InstanceNormalization(dim_out_stack, affine=False)]
    netG = nn.Sequential(*layers)
    netG.apply(weights_init)
    return netG

def define_D(nc, nf, imageSize, n_layers,  n_residual, has_sigmoid, dropout=0):
    stacks = np.log2(imageSize) - 1
    assert int(stacks) == stacks, 'Image not power of 2 : {}' % (imageSize)
    stacks = int(stacks)
    stack_dim = nf
    non_linearity = functools.partial(nn.LeakyReLU, negative_slope=.2)
    layers = []
    for stack in range(stacks):
        dim_in_stack = nc if stack == 0 else stack_dim
        dim_out_stack = stack_dim if stack == 0 else stack_dim * 2
        stack_dim = stack_dim if stack == 0 else stack_dim * 2
        layers += FeatureMapChanger(dim_in_stack, dim_out_stack, non_linearity=non_linearity, up=False)
        layers = layers + [nn.Dropout2d(p=dropout)] if dropout > 0 else layers
        layers += [ResnetBlock(dim_out_stack) for residual in range(n_residual)]
        for layer in range(n_layers):
            layers += [CoBlock(dim_out_stack, dim_out_stack, non_linearity=non_linearity)]
            layers = layers + [nn.Dropout2d(p=dropout)] if dropout > 0 else layers
            layers += [ResnetBlock(dim_out_stack) for residual in range(n_residual)]

    layers += [nn.Conv2d(dim_out_stack, 1, 2)]
    if has_sigmoid:
        layers += [nn.Sigmoid()]
    netD = nn.Sequential(*layers)
    netD.apply(weights_init)
    return netD

def define_D_latent(nz, nf, dropout, n_layers, has_sigmoid):
    layers = []
    for layer in range(1, n_layers + 1):
        if layer == 1:
            layers += [CoBlock(nz, nf, kernel_size=1, padding=0)]
            layers += [nn.Dropout2d(p=dropout)]
        elif layer == n_layers:
            layers += [CoBlock(nf, 1, kernel_size=1, padding=0)]
        else:
            layers += [CoBlock(nf, nf, kernel_size=1, padding=0)]
    if has_sigmoid:
        layers += [nn.Sigmoid()]
    netD = nn.Sequential(*layers)
    netD.apply(weights_init)
    return netD

def FeatureMapChanger(dim_in, dim_out, current_image_size=None, non_linearity=nn.ReLU, norm_layer=nn.BatchNorm2d, upsampling=True, up=True):
    layers = []
    if up and upsampling:
        layers += [nn.UpsamplingNearest2d(scale_factor=2)]
        layers += [CoBlock(dim_in, dim_out, norm_layer, non_linearity)]
    elif up:
        if current_image_size == 1:
            layers += [nn.ConvTranspose2d(dim_in, dim_out, 2, padding=0, stride=1)]
        else:
            layers += [nn.ConvTranspose2d(dim_in, dim_out, 4, padding=1, stride=2)]
        layers += [norm_layer(dim_out)]
        layers += [non_linearity(inplace=True)]
    else:
        layers += [CoBlock(dim_in, dim_out, norm_layer, non_linearity, stride=2)]
    return layers

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# Instance Normalization layer from
# https://github.com/darkstar112358/fast-neural-style
class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-5, affine=True):
        super(InstanceNormalization, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()
        self.affine = affine

    def _reset_parameters(self):
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        if n > 1:
            t = x.view(x.size(0), x.size(1), n)
        else:
            t = x.view(x.size(0), 1, x.size(1))
        mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).expand_as(x)
        if n > 1:
            var *= (n - 1) / float(n)
        out = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
            scale_broadcast = scale_broadcast.expand_as(x)
            shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
            shift_broadcast = shift_broadcast.expand_as(x)
            out = out * scale_broadcast + shift_broadcast
        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer)

    def build_conv_block(self, dim, norm_layer):
        conv_block = []

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       norm_layer(dim),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# Define a Convolution block
class CoBlock(nn.Module):
    def __init__(self, dim_in, dim_out, norm_layer=nn.BatchNorm2d, non_linearity=nn.ReLU, stride=1, kernel_size=3, padding=1):
        super(CoBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim_in, dim_out, kernel_size, padding, stride, norm_layer, non_linearity)

    def build_conv_block(self, dim_in, dim_out, kernel_size, padding, stride, norm_layer, non_linearity):
        conv_block = [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=padding, stride=stride)]
        conv_block += [norm_layer(dim_out)]
        conv_block += [non_linearity(inplace=True)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.conv_block(x)

# x -> Relu(x) + sign(x)
class NonLinearity(nn.Module):
    def __init__(self, inplace=False):
        super(NonLinearity, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            F.ReLU(x, inplace=True)
            x.add_(torch.sign(x))
        else:
            output = F.ReLU(x)
            return output + torch.sign(output)

# Add gaussian Noise
class Noiser(nn.Module):
    def __init__(self, noise, tensor):
        super(Noiser, self).__init__()
        self.noise = noise
        self.tensor = tensor

    def forward(self, x):
        return x.data.add(self.tensor(x.size()).normal_(0, self.noise))
