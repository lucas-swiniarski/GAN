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

import sys
sys.path.append("..")
import utils
import functools

# For printing in real time on HPC
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
# Global parameters :
parser.add_argument('--dataset', type=str, default='mnist',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
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
parser.add_argument('--clamp', action='store_true', help='Gan Discriminator')
parser.add_argument('--wasserstein', action='store_true', default=False, help='Training a Wasserstein Gan or a DC-GAN')
parser.add_argument('--c', type=float, default=0.01, help='Clamping parameter of the W-Gan')
parser.add_argument('--lbda', type=float, default=.3, help='Cycle loss factor')

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
sys.path.append('../models/DiscriminativeLatentNN')

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
        import GenMnist as ModelGImage
    elif args.model_g == 'upsampling':
        import GenMnistUpsampling as ModelGImage
    elif args.model_g == 'residual':
        import GenMnistResidual as ModelGImage
    import DiscLatent as ModelDLatent
    import DiscMnist as ModelMixed

    imageSize, nc = 28, 1

###
# Import data-sets.
###

trainloader, testloader, n_class = utils.load_dataset(args.dataset, args.dataroot, args.batchSize, imageSize, args.workers, args.training_size)

assert trainloader, testloader

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

netGImage = ModelGImage._netG(args.nz, args.ngf, nc, args.bng_momentum)
netGImage.apply(utils.weights_init)
print(netGImage)

netDImage = ModelMixed._netD(args.ndf, nc, args.wasserstein, False, 0, args.bias, args.dropout, args.bnd_momentum, 0)
netDImage.apply(utils.weights_init)
print(netDImage)

netGLatent = ModelMixed._netD(args.ndf, nc, args.wasserstein, False, 0, False, False, args.bng_momentum, 0, args.nz)
netGLatent.apply(utils.weights_init)
print(netGLatent)

netDLatent = ModelDLatent._netD(args.ndf, args.nz, args.wasserstein, args.bias, args.bnd_momentum)
netDLatent.apply(utils.weights_init)
print(netDLatent)

input = torch.FloatTensor(args.batchSize, 3, imageSize, imageSize)
latent = torch.FloatTensor(args.batchSize, args.nz, 1, 1)
# Fixed latent to plot generated images every 100 batches.
fixed_latent = utils.generate_latent_tensor(100, args.nz, args.noise_unconnex, 0, torch.range(0, 9).repeat(10).long(), 0)

if args.cuda:
    netGImage.cuda()
    netDImage.cuda()
    netGLatent.cuda()
    netDLatent.cuda()
    input = input.cuda()
    latent = latent.cuda()
    fixed_latent = fixed_latent.cuda()

input = Variable(input)
latent = Variable(latent)
fixed_latent = Variable(fixed_latent)

if args.adam:
    optimizerDLatent = optim.Adam(netDLatent.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
    optimizerDImage = optim.Adam(netDImage.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
    optimizerG = optim.Adam([{'params' : netGLatent.parameters()}, {'params' : netGImage.parameters()}], lr = args.lr, betas = (args.beta1, 0.999))
else:
    optimizerDLatent = optim.RMSprop(netDLatent.parameters(), lr = args.lr)
    optimizerDImage = optim.RMSprop(netDImage.parameters(), lr = args.lr)
    optimizerG = optim.RMSprop([{'params' : netGLatent.parameters()}, {'params' : netGImage.parameters()}], lr = args.lr)

if args.ngpu > 1:
    gpu_ids = range(ngpu)
    optimizerDLatent = torch.nn.DataParallel(optimizerDLatent, device_ids=gpu_ids)
    optimizerDImage = torch.nn.DataParallel(optimizerDImage, device_ids=gpu_ids)
    optimizerG = torch.nn.DataParallel(optimizerG, device_ids=gpu_ids)

# Keeps tracks of discriminator vs generator number of training
critic_trained_times = 0

for epoch in range(1, args.epochs + 1):
    for i, (data, _) in enumerate(trainloader, 0):

        batch_size = data.size(0)

        input.data.resize_(data.size()).copy_(data)
        if args.noise:
            input.data.add_(torch.FloatTensor(data.size()).normal_(0, .01))

        latent.data.resize_(batch_size, args.nz, 1, 1).copy_(utils.generate_latent_tensor(batch_size, args.nz, args.noise_unconnex, 0))

        fakeImage = netGImage(latent)
        fakeLatent = netGLatent(input)[0]
        fakeLatent.data.resize_(batch_size, args.nz, 1, 1)

        ############################
        # (1) Update D Image network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        netDImage.train()
        netDImage.zero_grad()

        # train with real

        output = netDImage(input)

        errD_I_real = 0

        if args.wasserstein:
            errD_I_real -= torch.mean(output[0])
        else:
            label_rf.data.resize_(batch_size).fill_(real_label)
            errD_I_real += criterion_rf(output[0], label_rf)
            errD_I_real.backward()

        D_I_x = output[0].data.mean()

        # train with fake

        output = netDImage(fakeImage.detach())

        errD_I_fake = 0

        if args.wasserstein:
            errD_I_fake += torch.mean(output[0])
        else:
            label_rf.data.fill_(fake_label)
            errD_I_fake += criterion_rf(output[0], label_rf)
            errD_I_fake.backward()
        D_I_z = output[0].data.mean()

        # Step
        errD_I = errD_I_real + errD_I_fake

        if args.wasserstein:
            errD_I.backward()

        optimizerDImage.step()

        if args.clamp:
            netDImage.apply(functools.partial(utils.weights_clamp, c=args.c))

        ############################
        # (2) Update D Latent network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        netDLatent.train()
        netDLatent.zero_grad()

        # train with real

        output = netDLatent(latent)

        errD_L_real = 0

        if args.wasserstein:
            errD_L_real -= torch.mean(output[0])
        else:
            label_rf.data.resize_(batch_size).fill_(real_label)
            errD_L_real += criterion_rf(output[0], label_rf)
            errD_L_real.backward()

        D_L_x = output[0].data.mean()

        # train with fake
        output = netDLatent(fakeLatent.detach())

        errD_L_fake = 0

        if args.wasserstein:
            errD_L_fake += torch.mean(output[0])
        else:
            label_rf.data.fill_(fake_label)
            errD_L_fake += criterion_rf(output[0], label_rf)
            errD_L_fake.backward()

        D_L_z = output[0].data.mean()

        # Step
        errD_L = errD_L_real + errD_L_fake

        if args.wasserstein:
            errD_L.backward()

        optimizerDLatent.step()

        if args.clamp:
            netDLatent.apply(functools.partial(utils.weights_clamp, c=args.c))

        critic_trained_times += 1

        ############################
        # (3) Update G networks :
        # - netGImages : maximize log(D(G(z)))
        # - netGLatent : maximize log(D(G(z)))
        # - Cycle Image : minimize E[||netGImages(netGLatent(x)) - x||]
        # - Cycle Latent : minimize E[||netGLatent(netGImages(z)) - z||]
        ###########################

        if critic_trained_times == args.n_critic:
            netDImage.train()
            netDLatent.train()
            critic_trained_times = 0

            netGImage.zero_grad()
            netGLatent.zero_grad()

            # netGImage disc. loss
            output = netDImage(fakeImage)

            errG_I = 0

            if args.wasserstein:
                errG_I += - torch.mean(output[0])
            else:
                label_rf.data.fill_(real_label) # fake labels are real for generator cost
                errG_I += criterion_rf(output[0], label_rf)

            # latent -> image -> latent loss
            output = netGLatent(fakeImage)
            circle_L = torch.mean(torch.pow(latent - output[0], 2))

            # netGLatent disc. loss
            output = netDLatent(fakeLatent)

            errG_L = 0

            if args.wasserstein:
                errG_L += - torch.mean(output[0])
            else:
                label_rf.data.fill_(real_label) # fake labels are real for generator cost
                errG_L += criterion_rf(output[0], label_rf)

            # image -> latent -> image loss
            output = netGImage(fakeLatent)
            circle_I = torch.mean(torch.abs(input - output))
            factor = args.lbda * epoch / args.epochs
            errG = errG_I + errG_L +  (circle_L + circle_I) * factor
            errG.backward()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_D_Latent : %.4f [D(x) : %.4f D(G(z)) : %.4f] Loss_D_Image : %.4f [D(x) : %.4f D(G(z)) : %.4f] Loss_G : %.4f [Img : %.4f Lat : %.4f] Circle : [Img : %.4f Lat : %.4f]'
                  % (epoch, args.epochs, i, len(trainloader),
                     errD_L.data[0], D_L_x, D_L_z,
                     errD_I.data[0], D_I_x, D_I_z,
                     errG.data[0], errG_I.data[0], errG_L.data[0],
                     circle_I.data[0], circle_L.data[0]))


        if i % 100 == 0:
            vutils.save_image(utils.mix(input.data, netGImage(fakeLatent).data),
                    '%s/%s_real_reconstruct_samples.png' % (args.outf, args.name))
            fake = netGImage(fixed_latent)
            vutils.save_image(fake.data,
                    '%s/%s_fake_samples_epoch_%03d.png' % (args.outf, args.name, epoch)
                    , nrow=10)

    # do checkpointing
    torch.save(netGImage.state_dict(), '%s/%s_netGImg_epoch_%d.pth' % (args.outf, args.name, epoch))
    torch.save(netGLatent.state_dict(), '%s/%s_netGLat_epoch_%d.pth' % (args.outf, args.name, epoch))
