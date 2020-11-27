"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import sys
import random
class OursModel(BaseModel):
    '''Our model'''
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned', norm='batch', netG='resnet_9blocks')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for the regression loss')  # You can define new arguments for this model.
            parser.add_argument('--lambda_C', type=float, default=2.0, help='weight for inter_iter loss.')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'inter_iter_', 'progress_']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real_A', 'fake_B', 'real_B',]#'D_maps']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G','D']
        else:
            self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc_G, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:  # only defined during training time
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.inter_iter = torch.nn.L1Loss()
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # if self.isTrain:
        self.num_iter = opt.num_iter
        self.fake_A_ = None
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.real_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.real_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths

    def forward(self, iter=-1):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # if self.isTrain:
        assert self.num_iter > 0
        # if self.fake_A_ is None:
        self.fake_A_ = torch.zeros_like(self.real_A)
            # print('using zero as fake A...once')
        # fake_B_ = torch.zeros_like(self.real_A)

        if self.isTrain:
            if iter == 0:
                # print(self.real_A.shape)
                # print(self.fake_A_.shape)
                in_ = torch.cat([self.real_A, self.fake_A_], 1)
            else:
                in_ = torch.cat([self.real_A, self.fake_B[-1]], 1)
            self.fake_B.append(self.netG(in_))  # generate output image given the input data_A
        else:
            if iter == 0:
                in_ = torch.cat([self.real_A, self.fake_A_], 1)  # G(A)
            else:
                in_ = torch.cat([self.real_A, self.fake_B], 1)
            self.fake_B = self.netG(in_)
    def backward_D(self, iter):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B[-1]), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self, iter):
        # print('iter {}'.format(iter))
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B[-1]), 1)
        self.D_maps.append(self.netD(fake_AB))
        self.loss_G_GAN = self.criterionGAN(self.D_maps[-1], True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B[-1], self.real_B) * self.opt.lambda_L1
        # combine loss and calculate gradients

        self.loss_inter_iter = 0.0
        self.loss_progress = 0.0
        if iter == 1 or iter == 2:
            self.loss_inter_iter = self.opt.lambda_C * self.inter_iter(self.fake_B[-1], self.fake_B[-2])
            self.loss_progress = 2 * torch.max(torch.zeros_like(self.D_maps[-1]).to(self.device),  -5e-4 + self.D_maps[-2]-self.D_maps[-1]).mean()
            'for logging'
            self.loss_progress_ = self.loss_progress.item()
            self.loss_inter_iter_ = self.loss_inter_iter.item()
        # self.loss_inter_iter_ = 0.0
        # self.loss_progress_ = 0.0
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_inter_iter + self.loss_progress
        # print('real a leaf {} requires_grad {}'.format(self.real_A.is_leaf, self.real_A.grad))
        # print('fake b leaf {} requires_grad {}'.format(self.fake_B[-1].is_leaf, self.fake_B[-1].grad))
        # print('real b leaf111 {} requires_grad111 {}'.format(self.real_B.is_leaf, self.real_B.grad))
        self.loss_G.backward()
        # print('real a leaf {} requires_grad {}'.format(self.real_A.is_leaf, self.real_A.requires_grad))
        # print('fake b leaf {} requires_grad {}'.format(self.fake_B[-1].is_leaf, self.fake_B[-1].requires_grad))
        # print('real b leaf {} requires_grad {}'.format(self.real_B.is_leaf, self.real_B.requires_grad))
        # self.fake_B[-1] = self.fake_B[-1].detach()
        # self.fake_B[-1].detach_()
        # self.fake_B[-1].retain_grad()
        # self.fake_B[-1] = self.fake_B[-1].detach()#.clone()
        # print('fake b 111leaf {} 111requires_grad {}'.format(self.fake_B[-1].is_leaf, self.fake_B[-1].requires_grad))

        # if not iter == 0:
        #     self.fake_B[-2] = self.fake_B[-2].detach()

        # self.real_A = self.real_A.detach()
    def optimize_parameters(self):
        self.fake_B = []
        self.D_maps =[]
        for iter in range(self.num_iter):
            self.forward(iter)  # compute fake images: G(A)
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D

            # print('G grad 0 {}'.format(self.print_grad(self.netG)))
            # print('D grad 0 {}'.format(self.print_grad(self.netD)))

            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D(iter)  # calculate gradients for D
            # print('G grad 5 {}'.format(self.print_grad(self.netG)))
            # print('D grad 5 {}'.format(self.print_grad(self.netD)))
            self.optimizer_D.step()  # update D's weights
            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            # print('G grad 10 {}'.format(self.print_grad(self.netG)))
            # print('D grad 10 {}'.format(self.print_grad(self.netD)))

            # print('G grad {}'.format(self.print_grad(self.netG)))
            # print('D grad {}'.format(self.print_grad(self.netD)))
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G(iter)  # calculate graidents for G
            # print('G grad 15 {}'.format(self.print_grad(self.netG)))
            # print('D grad 15 {}'.format(self.print_grad(self.netD)))
            self.optimizer_G.step()  # udpate G's weights
            self.fake_B[-1] = self.fake_B[-1].detach()
            self.D_maps[-1] = self.D_maps[-1].detach()
            # self.fake_A_ = random.choice(self.fake_B).clone()
        # sys.exit()
        # self.fake_B = None
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        if self.isTrain:
            self.fake_B = torch.cat([_ for _ in self.fake_B], 2)
            self.D_maps = torch.cat([_ for _ in self.D_maps], 2)

        # print('======>fake img size {}'.format(self.fake_B.shape))
    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            for iter in range(self.num_iter):
                self.forward(iter)
            self.compute_visuals()
