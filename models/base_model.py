import os
import torch
from pdb import set_trace as st

# From :
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/base_model.py

class BaseModel():
    def __init__(self, args):
        self.args = args
        self.tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
        self.save_dir = os.path.join(args.outf, args.name)
        self.old_lr = args.lr

    def forward(self, x):
        pass

    def save(self, label):
        pass

    def set_input(self, input):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if self.args.cuda and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass

    def __call__(self, x):
        return self.forward(x)
