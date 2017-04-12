from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

# For printing in real time on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

sys.path.append("..")
import utils
import functools

parser = argparse.ArgumentParser()
# Global parameters :
parser.add_argument('--dataset', type=str, default='mnist',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='../TrainedNetworks', help='folder to output images and model checkpoints')
parser.add_argument('--name', default='dcgan', help='Name used to save models')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--training-size', type=int, default=-1, help='How many examples of real data do we use, (default:-1 = Infinity)')

# Learning Parameters :
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--adam', action='store_true', help='Default RMSprop optimizer, wether to use Adam instead')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

# Generator Parameters :
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--bng-momentum', type=float, default=0.1, help='Momentum of BatchNorm Generator')
parser.add_argument('--model-g', type=str, default='base', help='Generator model : base | upsampling | residual')
parser.add_argument('--noise-unconnex', action='store_true', help='Train with an Unconnex Noise vector (No N(0,1))')

# Discriminator Parameters :
parser.add_argument('--noise', action='store_true', help='Add gaussian noise to real data')
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--bnd-momentum', type=float, default=0.1, help='Momentum of BatchNorm Discriminator')
parser.add_argument('--dropout', action='store_true', help='Dropouts on discriminator')
parser.add_argument('--bias', action='store_true', help='Bias term on convolutions on discriminator')

# Gan type :
parser.add_argument('--n-critic', type=int, default=1,help='Times training the discriminator vs generator')
parser.add_argument('--ac-gan', action='store_true', help='Use a class AC-GAN')
parser.add_argument('--info-gan-latent', type=int, default=0, help='Info-GAN latent space size (c in paper)')
parser.add_argument('--clamp', action='store_true', help='Gan Discriminator')
parser.add_argument('--wasserstein', action='store_true', default=False, help='Training a Wasserstein Gan or a DC-GAN')
parser.add_argument('--c', type=float, default=0.01, help='Clamping parameter of the W-Gan')

#TODO : Implement grad-pen when backprop of gradients enabled
parser.add_argument('--grad-pen', action='store_true', default=False, help='Penalize Gradient')
parser.add_argument('--reg-factor', type=float, default=10, help='Improved WGAN training')

args = parser.parse_args()

args.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

print(args)

###
# Import Generative and Disriminative Networks, depending on the data-set ( image size).
###

sys.path.append("../models/DiscriminativeNN")
sys.path.append("../models/GenerativeNN")

if args.dataset == 'cifar10':
    if args.model_g == 'base':
        import GenCifar10 as ModelG
    elif args.model_g == 'upsampling':
        import GenCifar10Upsampling as ModelG
    elif args.model_g == 'residual':
        import GenCifar10Residual as ModelG
    import DiscCifar10 as ModelD
    imageSize, nc = 32, 3
elif args.dataset == 'mnist':
    if args.model_g == 'base':
        import GenMnist as ModelG
    elif args.model_g == 'upsampling':
        import GenMnistUpsampling as ModelG
    elif args.model_g == 'residual':
        import GenMnistResidual as ModelG
    import DiscMnist as ModelD
    imageSize, nc = 28, 1

###
# Import data-sets.
###

trainloader, testloader, n_class = utils.load_dataset(args.dataset, args.dataroot, args.batchSize, imageSize, args.workers, args.training_size)

assert trainloader, testloader

###
# Initialize networks and useful variables depending on training loss ( Jensen-Shanon, Wasserstein, ... ).
###

if args.ac_gan:
    criterion_c = nn.CrossEntropyLoss()
    if args.cuda:
        criterion_c.cuda()
else:
    n_class = 0

if not args.wasserstein:
    criterion_rf = nn.BCELoss()
    label_rf = torch.FloatTensor(args.batchSize)
    label_rf = Variable(label_rf)
    real_label,fake_label = 1, 0
    if args.cuda:
        criterion_rf.cuda()
        label_rf = label_rf.cuda()

# The Input size of the generate
n_input = args.nz + n_class + args.info_gan_latent

netG = ModelG._netG(n_input, args.ngf, nc, args.bng_momentum)
netG.apply(utils.weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD = ModelD._netD(args.ndf, nc, args.wasserstein, args.ac_gan, n_class, args.bias, args.dropout, args.bnd_momentum, args.info_gan_latent)
netD.apply(utils.weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

# Real Data Images
input = torch.FloatTensor(args.batchSize, 3, imageSize, imageSize)
# Real Data Target Class
label_class = torch.LongTensor(args.batchSize)
latent = torch.FloatTensor(args.batchSize, n_input, 1, 1)
fixed_latent = utils.generate_latent_tensor(100, args.nz, args.noise_unconnex, n_class, torch.range(0, 9).repeat(10).long(), args.info_gan_latent)

if args.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    latent, fixed_latent, label_class = latent.cuda(), fixed_latent.cuda(), label_class.cuda()

input = Variable(input)
latent = Variable(latent)
fixed_latent = Variable(fixed_latent)
label_class = Variable(label_class)

if args.adam:
    optimizerD = optim.Adam(netD.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = args.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr = args.lr)

if args.info_gan_latent > 0:
    optimizerQ = optim.Adam([{'params': netG.parameters()},{'params': netD.parameters()}], lr=args.lr)

if args.ngpu > 1:
    gpu_ids = range(ngpu)
    optimizerD = torch.nn.DataParallel(optimizerD, device_ids=gpu_ids)
    optimizerG = torch.nn.DataParallel(optimizerG, device_ids=gpu_ids)
    if args.info_gan_latent > 0:
        optimizerQ = torch.nn.DataParallel(optimizerQ, device_ids=gpu_ids)

# Keeps tracks of discriminator vs generator number of training
critic_trained_times = 0

for epoch in range(1, args.epochs + 1):
    for i, (data, target) in enumerate(trainloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # train with real
        netD.train()
        netD.zero_grad()

        batch_size = data.size(0)
        if args.noise:
            eps = torch.FloatTensor(data.size()).normal_(0, 1.0 / 256)
            input.data.resize_(data.size()).copy_(data.add(eps))
        else:
            input.data.resize_(data.size()).copy_(data)
        label_class.data.resize_(batch_size).copy_(target)

        # Output is a list of :
        # - Real/Fake output
        # - Auxiliary Classifier GAN class, if exists
        # - InfoGAN parameter, if exists
        output = netD(input,)

        errD_real = 0
        if args.ac_gan:
            loss_C = criterion_c(output[1], label_class)
            errD_real += loss_C
        if args.info_gan_latent > 0:
            loss_Info = 0 # Implement Loss with output[-1]
            errD_real += loss_Info

        if args.wasserstein:
            errD_real -= torch.mean(output[0])
        else:
            label_rf.data.resize_(batch_size).fill_(real_label)
            errD_real += criterion_rf(output[0], label_rf)
            errD_real.backward()

        D_x = output[0].mean().detach()

        # train with fake

        latent.data.resize_(batch_size, args.nz + n_class + args.info_gan_latent, 1, 1).copy_(utils.generate_latent_tensor(batch_size, args.nz, args.noise_unconnex, n_class, target, args.info_gan_latent))
        fake = netG(latent)

        #TODO: When Backprop through gradient implemented, here should be the augmented wasserstein paper Implementation

        output =netD(fake.detach())

        errD_fake = 0

        # Should we use this ? This would be the exact AC-GAN paper :
        # if args.ac_gan:
        #     errD_fake += criterion_c(output[1], label_class)
        #     loss_C += criterion_c(output[1], label_class)

        if args.wasserstein:
            errD_fake += torch.mean(output[0])
        else:
            label_rf.data.fill_(fake_label)
            errD_fake += criterion_rf(output[0], label_rf)
            errD_fake.backward()
        D_G_z1 = output[0].data.mean()

        # Step
        errD = errD_real + errD_fake

        if args.wasserstein:
            errD.backward()

        optimizerD.step()

        if args.clamp:
            netD.apply(functools.partial(utils.weights_clamp, c=args.c))
        critic_trained_times += 1

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        if critic_trained_times == args.n_critic:
            # Note : We used to choose wether eval or not the disc. But it never works when on eval mode.
            netD.train()
            critic_trained_times = 0

            netG.zero_grad()

            output = netD(fake)

            errG = 0

            if args.ac_gan:
                loss_C_G = criterion_c(output[1], label_class)
                errG += loss_C_G
            if args.wasserstein:
                errG += - torch.mean(output[0])
            else:
                label_rf.data.fill_(real_label) # fake labels are real for generator cost
                errG += criterion_rf(output[0], label_rf)

            errG.backward()
            D_G_z2 = output[0].data.mean()
            optimizerG.step()

        ############################
        # (3) Update Q network if InfoGAN
        ############################
        errQ = Variable(torch.FloatTensor([0]))

        if args.info_gan_latent > 0:
            netD.zero_grad()
            netG.zero_grad()

            fake = netG(latent)
            output = netD(fake)

            c = utils.get_latent_code(latent, args.info_gan_latent).squeeze()
            crossent_loss = torch.mean(torch.sum(torch.exp( - c ** 2 / 2) * torch.log(output[-1] + 1e-8), dim=1))
            errQ =  - args.reg_factor * crossent_loss
            errQ.backward()
            optimizerQ.step()

        if args.ac_gan:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f Loss_G : %.4f Loss_G_C : %.2f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, i, len(trainloader),
                     errD.data[0] - loss_C.data[0], loss_C.data[0], errG.data[0] - loss_C_G.data[0], loss_C_G.data[0],D_x.data[0], D_G_z1, D_G_z2))
        else:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_Q : %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, i, len(trainloader),
                     errD.data[0], errG.data[0], errQ.data[0], D_x.data[0], D_G_z1, D_G_z2))


        if i % 100 == 0:
            vutils.save_image(data,
                    '%s/%s_real_samples.png' % (args.outf, args.name))
            fake = netG(fixed_latent)
            vutils.save_image(fake.data,
                    '%s/%s_fake_samples_epoch_%03d.png' % (args.outf, args.name, epoch)
                    , nrow=10)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/%s_netG_epoch_%d.pth' % (args.outf, args.name, epoch))
    torch.save(netD.state_dict(), '%s/%s_netD_epoch_%d.pth' % (args.outf, args.name, epoch))
