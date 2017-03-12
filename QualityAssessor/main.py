from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

# For printing in real time on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', required=True, type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='../TrainedNetworks', help='folder to output images and model checkpoints')
parser.add_argument('--name', default='dcgan', help='Name of the saved modle')
parser.add_argument('--Wasserstein', type=bool, default=False, help='Training a Wasserstein Gan or a DC-GAN')
parser.add_argument('--clamp', type=bool, default=False, help='Do we clamp ? - Wasserstein Gan Discriminator')
parser.add_argument('--c', type=float, default=0.01, help='Clamping parameter of the W-Gan')
parser.add_argument('--n-critic', type=int, required=True, help='Times training the discriminator vs generator')

args = parser.parse_args()

args.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

###
# Import Generative and Disriminative Networks.
###

sys.path.append("../DiscriminativeNN")
sys.path.append("../GenerativeNN")

if args.dataset == 'cifar10':
    import GenCifar10 as ModelG
    import DiscCifar10 as ModelD
    if args.imageSize != 32:
        print('Model do not work with this image size !')
elif args.dataset == 'mnist':
    import GenMnist as ModelG
    import DiscMnist as ModelD
    if args.imageSize != 28:
        print('Model do not work with this image size !')
print(args)

if args.dataset == 'cifar10':
    transform = transforms.Compose([
        transforms.Scale(args.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = dset.CIFAR10(root=args.dataroot, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))

    testset = dset.CIFAR10(root=args.dataroot, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=int(args.workers))
elif args.dataset == 'mnist':
    transform = transforms.Compose([
        transforms.Scale(args.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = dset.MNIST(root=args.dataroot, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize, shuffle=True, num_workers=int(args.workers))

    testset = dset.MNIST(root=args.dataroot, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize, shuffle=False, num_workers=int(args.workers))

assert trainloader, testloader

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)

if args.dataset == 'mnist':
    nc = 1
else:
    nc = 3

netG = ModelG._netG(ngpu, nz, ngf, nc)
netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD = ModelD._netD(ngpu, nz, ndf, nc, args.Wasserstein)
netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
noise = torch.FloatTensor(args.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_(0, 1)

if args.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)

noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

if args.Wasserstein:
    optimizerD = optim.RMSprop(netD.parameters(), lr = args.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr = args.lr)
else:
    optimizerD = optim.Adam(netD.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
    criterion = nn.BCELoss()
    label = torch.FloatTensor(args.batchSize)
    real_label = 1
    fake_label = 0
    label = Variable(label)
    if args.cuda:
        criterion.cuda()
        label = label.cuda()

critic_trained_times = 0

for epoch in range(args.niter):
    for i, data in enumerate(trainloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        output = netD(input)
        if args.Wasserstein:
            errD_real = torch.mean(output)
        else:
            label.data.resize_(batch_size).fill_(real_label)
            errD_real = criterion(output, label)
            errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise)
        output = netD(fake.detach())
        if args.Wasserstein:
            errD_fake = torch.mean(output)
        else:
            label.data.fill_(fake_label)
            errD_fake = criterion(output, label)
            errD_fake.backward()
        D_G_z1 = output.data.mean()

        if args.Wasserstein:
            errD = - errD_real + errD_fake
            errD.backward()
        else:
            errD = errD_real + errD_fake
        optimizerD.step()

        critic_trained_times += 1

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        if critic_trained_times == args.n_critic:
            critic_trained_times = 0
            netG.zero_grad()
            output = netD(fake)
            if args.Wasserstein:
                errG = - torch.mean(output)
            else:
                label.data.fill_(real_label) # fake labels are real for generator cost
                errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()
            if args.clamp:
                for p in netD.parameters():
                    p.data.clamp_(-args.c, args.c)

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.niter, i, len(trainloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/%s_real_samples.png' % (args.outf, args.name))
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/%s_fake_samples_epoch_%03d.png' % (args.outf, args.name, epoch))

    # do checkpointing
    torch.save(netG.state_dict(), '%s/%s_netG_epoch_%d.pth' % (args.outf, args.name, epoch))
    torch.save(netD.state_dict(), '%s/%s_netD_epoch_%d.pth' % (args.outf, args.name, epoch))
