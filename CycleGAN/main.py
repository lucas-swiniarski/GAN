from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

import sys
sys.path.append("..")
import utils
import functools

sys.path.append("../models")
import networks

# For printing in real time on HPC
# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
# Global parameters :
parser.add_argument('--dataset', type=str, default='mnist',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=32, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--outf', default='../TrainedNetworks', help='folder to output images and model checkpoints')
parser.add_argument('--name', default='dcgan', help='Name used to save models')
# parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--training-size', type=int, default=-1, help='How many examples of real data do we use, (default:-1 = Infinity)')

# Learning Parameters :
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--adam', action='store_true', help='Default RMSprop optimizer, wether to use Adam instead')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

# Image Generator Parameters :
parser.add_argument('--ngif', type=int, default=64)
parser.add_argument('--gi-upsampling', action='store_true', help='Upsampling vs. Deconvolution')
parser.add_argument('--gi-n-residual', type=int, default=0)
parser.add_argument('--gi-n-layers', type=int, default=0, help='Convolution Layers between Upsampling layers')
parser.add_argument('--gi-discontinuity', action='store_true', help='G with discontinuity')
parser.add_argument('--gi-in', action='store_true', help='G with InstanceNormalization instead of Batch Normalization')
parser.add_argument('--gi-noise', type=float, default=0, help='Gaussian Noise at each layer')
parser.add_argument('--gi-dropout', type=float, default=0, help='Dropout at each layer')

# Latent Generator Parameters :
parser.add_argument('--nglf', type=int, default=64)
parser.add_argument('--gl-upsampling', action='store_true', help='Upsampling vs. Deconvolution')
parser.add_argument('--gl-n-residual', type=int, default=0)
parser.add_argument('--gl-n-layers', type=int, default=0, help='Convolution Layers between Upsampling layers')
parser.add_argument('--gl-discontinuity', action='store_true', help='G with discontinuity')
parser.add_argument('--gl-in', action='store_true', help='G with InstanceNormalization instead of Batch Normalization')
parser.add_argument('--gl-noise', type=float, default=0, help='Gaussian Noise at each layer')
parser.add_argument('--gl-dropout', type=float, default=0, help='Dropout at each layer')

# Discriminator Image Parameters :
parser.add_argument('--ndif', type=int, default=64)
parser.add_argument('--di-n-layers', type=int, default=0)
parser.add_argument('--di-n-residual', type=int, default=0)
parser.add_argument('--di-dropout',  type=float, default=.2, help='Dropouts on discriminator')
parser.add_argument('--di-noise',  type=float, default=.2, help='Dropouts on discriminator')

# Discriminator Latent Parameters :
parser.add_argument('--ndlf', type=int, default=64)
parser.add_argument('--dl-n-layers', type=int, default=4)
parser.add_argument('--dl-dropout', type=float, default=.2, help='Dropouts on Latent Discriminator')

# Gan type :
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n-critic', type=int, default=1,help='Times training the discriminator vs generator')
parser.add_argument('--clamp', action='store_true', help='Gan Discriminator')
parser.add_argument('--wasserstein', action='store_true', default=False, help='Training a Wasserstein Gan or a DC-GAN')
parser.add_argument('--c', type=float, default=0.01, help='Clamping parameter of the W-Gan')
parser.add_argument('--lbda', type=float, default=.3, help='Cycle loss factor')

#TODO : Implement grad-pen when backprop of gradients enabled
parser.add_argument('--grad-pen', action='store_true', default=False, help='Penalize Gradient')
parser.add_argument('--reg-factor', type=float, default=10, help='Improved WGAN training')

args = parser.parse_args()

# fix seed
args.manualSeed = 1
print("Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

print(args)

###
# Import data-sets.
###

trainloader, nc = utils.load_dataset(args)
args.nc = nc
args.tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor

assert trainloader

###
# Initialize networks and useful variables depending on training loss ( Jensen-Shanon, Wasserstein, ... ).
###

if not args.wasserstein:
    # Standard GAN tools
    criterion_rf = nn.BCELoss()
    label_rf = torch.FloatTensor(args.batchSize)
    label_rf = Variable(label_rf)
    real_label,fake_label = 1, 0
    if args.cuda:
        criterion_rf.cuda()
        label_rf = label_rf.cuda()

from models.create_models import create_model

netGImage = create_model(args, 'g_image')
netDImage = create_model(args, 'd_image')
netGLatent = create_model(args, 'g_latent')
netDLatent = create_model(args, 'd_latent')

input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
latent = torch.FloatTensor(args.batchSize, args.nz, 1, 1)
# Fixed latent to plot generated images every 100 batches.
fixed_latent = torch.FloatTensor(100, args.nz, 1, 1)

if args.cuda:
    input = input.cuda()
    latent = latent.cuda()
    fixed_latent = fixed_latent.cuda()

input = Variable(input)
latent = Variable(latent)
fixed_latent = Variable(fixed_latent)

# Keeps tracks of discriminator vs generator number of training
critic_trained_times = 0

for epoch in range(1, args.epochs + 1):
    for i, (data, _) in enumerate(trainloader, 0):

        batch_size = data.size(0)

        input.data.resize_(data.size()).copy_(data)
        if args.di_noise > 0:
            noise = args.tensor(data.size()).normal_(0, args.di_noise)
            if args.cuda:
                noise = noise.cuda()
            input.data.add_(noise)
        latent.data.resize_(batch_size, args.nz, 1, 1).normal_(0, 1)

        fakeImage = netGImage(latent)
        fakeLatent = netGLatent(input)
        fakeLatent.data.resize_(batch_size, args.nz, 1, 1)
        
        ############################
        # (1) Update D Image network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        netDImage.zero_grad()

        # train with real

        output = netDImage(input)

        if args.wasserstein:
            errD_I_real = - torch.mean(output)
        else:
            label_rf.data.resize_(batch_size).fill_(real_label)
            errD_I_real = criterion_rf(output, label_rf)

        D_I_x = output.data.mean()


        # train with fake
        output = netDImage(fakeImage.detach())

        if args.wasserstein:
            errD_I_fake = torch.mean(output)
        else:
            label_rf.data.fill_(fake_label)
            errD_I_fake = criterion_rf(output, label_rf)

        D_I_z = output.data.mean()

        # Step
        errD_I = errD_I_real + errD_I_fake
        optimizerDImage.zero_grad()
        errD_I.backward()
        optimizerDImage.step()

        if args.clamp:
            netDImage.apply(functools.partial(networks.weights_clamp, c=args.c))

        ############################
        # (2) Update D Latent network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # train with real
        output = netDLatent(latent)
        D_L_x = output.data.mean()

        if args.wasserstein:
            errD_L_real = - torch.mean(output)
        else:
            label_rf.data.resize_(batch_size).fill_(real_label)
            errD_L_real = criterion_rf(output, label_rf)

        # train with fake
        output = netDLatent(fakeLatent.detach())
        D_L_z = output.data.mean()

        if args.wasserstein:
            errD_L_fake = torch.mean(output)
        else:
            label_rf.data.fill_(fake_label)
            errD_L_fake = criterion_rf(output, label_rf)

        # Step
        errD_L = errD_L_real + errD_L_fake
        optimizerDLatent.zero_grad()
        errD_L.backward()
        optimizerDLatent.step()

        if args.clamp:
            netDLatent.apply(functools.partial(networks.weights_clamp, c=args.c * 10))


        critic_trained_times += 1

        ############################
        # (3) Update G networks :
        # - netGImages : maximize log(D(G(z)))
        # - netGLatent : maximize log(D(G(z)))
        # - Cycle Image : minimize E[||netGImages(netGLatent(x)) - x||]
        # - Cycle Latent : minimize E[||netGLatent(netGImages(z)) - z||]
        ###########################

        if critic_trained_times == args.n_critic:
            critic_trained_times = 0

            # netGImage disc. loss
            output = netDImage(fakeImage)

            if args.wasserstein:
                errG_I = - torch.mean(output)
            else:
                label_rf.data.fill_(real_label) # fake labels are real for generator cost
                errG_I = criterion_rf(output, label_rf)

            # latent -> image -> latent loss
            # output = netGLatent(fakeImage)
            # circle_L = torch.mean(torch.pow(latent - output[0], 2))
            circle_L = Variable(torch.FloatTensor([0]))

            # netGLatent disc. loss
            output = netDLatent(fakeLatent)

            if args.wasserstein:
                errG_L = - torch.mean(output)
            else:
                label_rf.data.fill_(real_label) # fake labels are real for generator cost
                errG_L = criterion_rf(output, label_rf)

            # image -> latent -> image loss
            output = netGImage(fakeLatent)
            circle_I = torch.mean(torch.abs(input - output))
            errG = errG_I + errG_L +  circle_I * args.lbda
            optimizerG.zero_grad()
            errG.backward()
            optimizerG.step()
            print('Fake : [Mean : %.4f Std : %.4f] Real : [Mean : %.4f Std : %.4f] ' % (fakeLatent.data.mean(), fakeLatent.data.std(dim=1).mean(), latent.data.mean(), latent.data.std(dim=1).mean()))
            print('[%d/%d][%d/%d] Loss_D_Latent : %.4f [D(x) : %.4f D(G(z)) : %.4f] Loss_D_Image : %.4f [D(x) : %.4f D(G(z)) : %.4f] Loss_G : %.4f [Img : %.4f Lat : %.4f] Circle : [Img : %.4f]'
                  % (epoch, args.epochs, i, len(trainloader),
                     errD_L.data[0], D_L_x, D_L_z,
                     errD_I.data[0], D_I_x, D_I_z,
                     errG.data[0], errG_I.data[0], errG_L.data[0],
                     circle_I.data[0]))


        if i % 100 == 0:
            vutils.save_image(utils.mix(input.data, netGImage(fakeLatent).data),
                    '%s/%s_real_reconstruct_samples_epoch_%03d.png' % (args.outf, args.name, epoch))
            fake = netGImage(fixed_latent)
            vutils.save_image(fake.data,
                    '%s/%s_fake_samples_epoch_%03d.png' % (args.outf, args.name, epoch)
                    , nrow=10)

    # do checkpointing
    torch.save(netGImage.state_dict(), '%s/%s_netGImg_epoch_%d.pth' % (args.outf, args.name, epoch))
    torch.save(netGLatent.state_dict(), '%s/%s_netGLat_epoch_%d.pth' % (args.outf, args.name, epoch))
