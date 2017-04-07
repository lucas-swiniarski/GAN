import torch
from torch.autograd import Variable

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def sample_G(netG, nz, n=1):
    noise_tensor = torch.FloatTensor(n, nz)
    noise_tensor = Variable(noise_tensor)
    noise_tensor.data.resize_(n, nz)
    noise_tensor.data.normal_(0, 1)
    return netG(noise_tensor).data
