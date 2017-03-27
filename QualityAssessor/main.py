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
import utils

# For printing in real time on HPC
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10',help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--ac-gan', type=bool, default=False, help='Use a class AC-GAN')
parser.add_argument('--dataroot', default='../data', type=str, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', required=True, type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='../TrainedNetworks', help='folder to output images and model checkpoints')
parser.add_argument('--name', default='dcgan', help='Name of the saved modle')
parser.add_argument('--wasserstein', type=bool, default=False, help='Training a Wasserstein Gan or a DC-GAN')
parser.add_argument('--clamp', type=bool, default=False, help='Do we clamp ? - Wasserstein Gan Discriminator')
parser.add_argument('--c', type=float, default=0.01, help='Clamping parameter of the W-Gan')
parser.add_argument('--n-critic', type=int, required=True, help='Times training the discriminator vs generator')
parser.add_argument('--bias', type=bool, default=False, help='Bias term on convolutions on discriminator')
parser.add_argument('--dropout', type=bool, default=False, help='Dropouts on discriminator')
parser.add_argument('--clamping-method', type=str, default='clamp',help='clamp | normalize | max_normalize')
parser.add_argument('--noise', type=bool, default=False, help='Add gaussian noise to real data')
parser.add_argument('--training-size', type=int, default=-1, help='How many examples of real data do we use, (default:-1 = Infinity)')
parser.add_argument('--eval', type=bool, default=False, help='Do we train the Generator on Eval mode')
parser.add_argument('--bng-momentum', type=float, default=0.1, help='Momentum of BatchNorm Generator')
parser.add_argument('--bnd-momentum', type=float, default=0.1, help='Momentum of BatchNorm Discriminator')
parser.add_argument('--model-g', type=str, default='base', help='Generator model : base | upsampling | residual')
args = parser.parse_args()

args.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

###
# Import Generative and Disriminative Networks, depending on the data-set ( image size).
###

sys.path.append("../DiscriminativeNN")
sys.path.append("../GenerativeNN")

if args.dataset == 'cifar10':
    if args.model_g == 'base':
        import GenCifar10 as ModelG
    elif args.model_g == 'upsampling':
        import GenCifar10Upsampling as ModelG
    elif args.model_g == 'residual':
        import GenCifar10Residual as ModelG
    import DiscCifar10 as ModelD
    nc = 3
    if args.imageSize != 32:
        print('Model do not work with this image size !')
elif args.dataset == 'mnist':
    if args.model_g == 'base':
        import GenMnist as ModelG
    elif args.model_g == 'upsampling':
        import GenMnistUpsampling as ModelG
    elif args.model_g == 'residual':
        import GenMnistResidual as ModelG
    import DiscMnist as ModelD
    nc = 1
    if args.imageSize != 28:
        print('Model do not work with this image size !')
print(args)

###
# Import data-sets.
###

trainloader, testloader, n_class = utils.load_dataset(args.dataset, args.dataroot, args.batchSize, args.imageSize, args.workers, args.training_size)

assert trainloader, testloader

###
# Initialize networks and useful variables depending on training loss ( Jensen-Shanon, Wasserstein ).
###

ngpu = int(args.ngpu)
nz = int(args.nz)
n_input = nz
if args.ac_gan:
    n_input += n_class
ngf = int(args.ngf)
ndf = int(args.ndf)

netG = ModelG._netG(n_input, ngf, nc, args.bng_momentum)
netG.apply(utils.weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD = ModelD._netD(ndf, nc, args.wasserstein, args.ac_gan, n_class, args.bias, args.dropout, args.bnd_momentum)
netD.apply(utils.weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

input = torch.FloatTensor(args.batchSize, 3, args.imageSize, args.imageSize)
label_class = torch.LongTensor(args.batchSize)

if args.ac_gan:
    latent = torch.FloatTensor(args.batchSize, nz + n_class, 1, 1)
else:
    latent = torch.FloatTensor(args.batchSize, nz, 1, 1)

fixed_latent = torch.FloatTensor(100, nz, 1, 1).normal_(0, 1)

if args.ac_gan:
    criterion_c = nn.CrossEntropyLoss()
    if args.cuda:
        criterion_c.cuda()
    fixed_latent = utils.generate_latent_tensor(100, nz, n_class, torch.range(0, 9).repeat(10).long())
    fixed_latent.resize_(100, nz + n_class, 1, 1)

if args.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    latent, fixed_latent, label_class = latent.cuda(), fixed_latent.cuda(), label_class.cuda()

input = Variable(input)
latent = Variable(latent)
fixed_latent = Variable(fixed_latent)
label_class = Variable(label_class)

if args.wasserstein:
    optimizerD = optim.RMSprop(netD.parameters(), lr = args.lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr = args.lr)
else:
    optimizerD = optim.Adam(netD.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr = args.lr, betas = (args.beta1, 0.999))
    criterion_rf = nn.BCELoss()
    label_rf = torch.FloatTensor(args.batchSize)
    real_label = 1
    fake_label = 0
    label_rf = Variable(label_rf)
    if args.cuda:
        criterion_rf.cuda()
        label_rf = label_rf.cuda()

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

        if args.ac_gan:
            output_rf, output_c = utils.parallel_forward(netD, input, ngpu)
        else:
            output_rf = utils.parallel_forward(netD, input, ngpu)

        errD_real = 0
        if args.ac_gan:
            loss_C = criterion_c(output_c, label_class)
            errD_real += loss_C

        if args.wasserstein:
            errD_real -= torch.mean(output_rf)
        else:
            label_rf.data.resize_(batch_size).fill_(real_label)
            errD_real += criterion_rf(output_rf, label_rf)
            errD_real.backward()
        D_x = output_rf.data.mean()

        # train with fake

        if args.ac_gan:
            latent.data.resize_(batch_size, nz + n_class, 1, 1).copy_(utils.generate_latent_tensor(batch_size, nz, n_class, target))
        else:
            latent.data.resize_(batch_size, nz, 1, 1)
            latent.data.normal_(0, 1)

        fake = utils.parallel_forward(netG, latent, ngpu)
        if args.ac_gan:
            output_rf, output_c = utils.parallel_forward(netD, fake.detach(), ngpu)
        else:
            output_rf = utils.parallel_forward(netD, fake.detach(), ngpu)

        errD_fake = 0
        # Should we use this ?
        # if args.ac_gan:
        #     errD_fake += criterion_c(output_c, label_class)
        #     loss_C += criterion_c(output_c, label_class)

        if args.wasserstein:
            errD_fake += torch.mean(output_rf)
        else:
            label_rf.data.fill_(fake_label)
            errD_fake += criterion_rf(output_rf, label_rf)
            errD_fake.backward()
        D_G_z1 = output_rf.data.mean()

        # Step

        if args.wasserstein:
            errD = errD_real + errD_fake
            errD.backward()
        else:
            errD = errD_real + errD_fake
        optimizerD.step()

        critic_trained_times += 1

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        if critic_trained_times == args.n_critic:
            if args.eval:
                netD.eval()
            else:
                netD.train()
            critic_trained_times = 0
            netG.zero_grad()

            if args.ac_gan:
                output_rf, output_c = utils.parallel_forward(netD, fake, ngpu)
            else:
                output_rf = utils.parallel_forward(netD, fake, ngpu)

            errG = 0
            if args.ac_gan:
                loss_C_G = criterion_c(output_c, label_class)
                errG += loss_C_G
            if args.wasserstein:
                errG += - torch.mean(output_rf)
            else:
                label_rf.data.fill_(real_label) # fake labels are real for generator cost
                errG += criterion_rf(output_rf, label_rf)

            errG.backward()
            D_G_z2 = output_rf.data.mean()
            optimizerG.step()
            if args.clamp:
                netD.clamp(args.c, args.clamping_method)

            if args.ac_gan:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_C: %.4f Loss_G : %.4f Loss_G_C : %.2f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(trainloader),
                         errD_real.data[0] - loss_C.data[0], loss_C.data[0], errG.data[0] - loss_C_G.data[0], loss_C_G.data[0],D_x, D_G_z1, D_G_z2))
            else:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(trainloader),
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
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
