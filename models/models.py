import torch
import torch.nn as nn
from base_model import BaseModel
import networks
import torch.optim as optim

class _netGImage(BaseModel):
    def initialize(self, args):
        non_linearity = networks.NonLinearity if args.gi_discontinuity else nn.ReLU
        norm = networks.InstanceNormalization if args.gi_in else nn.BatchNorm2d
        self.model = networks.define_G(args.nz, args.nc, args.ngif, args.imageSize, args.gi_n_layers,
                                            args.gi_n_residual, self.tensor, args.gi_upsampling, norm=norm,
                                            non_linearity=non_linearity, noise=args.gi_noise, dropout=args.gi_dropout, up_to=2, up=True)
        networks.print_network(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr, betas = (args.beta1, 0.999)) if args.adam else optim.RMSprop(self.model.parameters(), lr = args.lr)
        if args.cuda:
            self.model.cuda()

    def forward(self, input):
        return self.model(input)

    def save(self, label):
        save_network(self.model, 'G_img', label)

    def update_learning_rate(self):
        lr = self.old_lr * self.args.niter_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class _netGLatent(BaseModel):
    def initialize(self, args):
        non_linearity = networks.NonLinearity if args.gl_discontinuity else nn.ReLU
        norm = networks.InstanceNormalization if args.gl_in else nn.BatchNorm2d
        self.model = networks.define_G(args.nz, args.nc, args.nglf, args.imageSize, args.gl_n_layers,
                                            args.gl_n_residual, self.tensor, args.gl_upsampling, norm=norm,
                                            non_linearity=non_linearity, noise=args.gl_noise, dropout=args.gl_dropout, up_to=2, up=False)
        networks.print_network(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr, betas = (args.beta1, 0.999)) if args.adam else optim.RMSprop(self.model.parameters(), lr = args.lr)
        if args.cuda:
            self.model.cuda()

    def forward(self, input):
        return self.model(input)

    def save(self, label):
        save_network(self.model, 'G_latent', label)

    def update_learning_rate(self):
        lr = self.old_lr * self.args.niter_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class _netDImage(BaseModel):
    def initialize(self, args):
        self.model = networks.define_D(args.nc, args.ndif, args.imageSize, args.di_n_layers,  args.di_n_residual, not args.wasserstein, args.di_dropout)
        networks.print_network(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr, betas = (args.beta1, 0.999)) if args.adam else optim.RMSprop(self.model.parameters(), lr = args.lr)
        if args.cuda:
            self.model.cuda()

    def forward(self, input):
        return self.model(input)

    def save(self, label):
        save_network(self.model, 'G_img', label)

    def update_learning_rate(self):
        lr = self.old_lr * self.args.niter_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class _netDLatent(BaseModel):
    def initialize(self, args):
        self.model = networks.define_D_latent(args.nz, args.ndlf, args.dl_dropout, args.dl_n_layers, not args.wasserstein)
        networks.print_network(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr = args.lr, betas = (args.beta1, 0.999)) if args.adam else optim.RMSprop(self.model.parameters(), lr = args.lr)
        if args.cuda:
            self.model.cuda()

    def forward(self, input):
        return self.model(input)

    def save(self, label):
        save_network(self.model, 'G_img', label)

    def update_learning_rate(self):
        lr = self.old_lr * self.args.niter_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
