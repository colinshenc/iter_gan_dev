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
import itertools
from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks
import torch.nn as nn
from torch.autograd import Variable
import sys
from collections import OrderedDict

class OursCycleGANModel(BaseModel):
    '''Our model'''

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_C', type=float, default=2.0, help='weight for inter_iter loss.')

            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser


    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'inter_iter_', 'progress_']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A','D_A_maps']
        visual_names_B = ['real_B', 'fake_A', 'rec_B','D_B_maps']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc_G, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nc_G, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.fake_A_pool = ImagePool(200)  # create image buffer to store previously generated images
        # self.fake_B_pool = ImagePool(200)  # create image buffer to store previously generated images
        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)


            # if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            #     assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            '''add inter-iter loss'''
            self.inter_iter_loss = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.lr = opt.lr
            self.beta1 = opt.beta1
            # self.beta1 = opt.beta1
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        self.bs = opt.batch_size
        self.num_iter = opt.num_iter
        assert self.num_iter > 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        self.real_A = input['A' if AtoB else 'B'].to(self.device)  # get image data A
        self.real_B = input['B' if AtoB else 'A'].to(self.device)  # get image data B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # get image paths


        #
        # self.optimizer_G.add_param_group({'params':self.fake_B,},)
        # self.optimizer_D.add_param_group({'params':self.fake_B,},)



    def forward(self, iter=-1):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # fake_B_ = self.fake_B_pool.get(bs=self.bs)
        # if fake_B_ is None:
        #      print('Using fake B none...once')
        fake_B_ = torch.zeros_like(self.real_A)

        # fake_A_ = self.fake_A_pool.get(bs=self.bs)
        # if fake_A_ is None:
        #      print('Using fake A none...once')
        fake_A_ = torch.zeros_like(self.real_A)

        if self.isTrain:
            if iter == 0:
                '''G*F,F*G'''
                # # print('fake b shape {}'.format(self.fake_B[-1].shape))
                # self.rec_A=self.netG_B(torch.cat([self.fake_B, fake_A_], 1))
                # self.fake_A=self.netG_B(torch.cat([self.real_B, fake_A_], 1))
                # self.rec_B=self.netG_A(torch.cat([self.fake_A, fake_B_], 1))

                self.fake_B.append(self.netG_A(torch.cat([self.real_A, fake_B_], 1)))
                # print('fake b shape {}'.format(self.fake_B[-1].shape))
                self.rec_A.append(self.netG_B(torch.cat([self.fake_B[-1], self.real_A], 1)))
                self.fake_A.append(self.netG_B(torch.cat([self.real_B, fake_A_], 1)))
                self.rec_B.append(self.netG_A(torch.cat([self.fake_A[-1], self.real_B], 1)))

                # self.optimizer_G.add_param_group({'params':self.fake_B,},)
                # self.optimizer_D.add_param_group({'params':self.fake_B,},)
            else:
                '''F2*G1,G2*F1; equivalent to G2*F1,F2*G1'''
                self.fake_B.append(self.netG_A(torch.cat([self.real_A, self.fake_B[-1]], 1))) #G1
                self.rec_A.append(self.netG_B(torch.cat([self.fake_B[-1], self.rec_A[-1]], 1)))  #F2
                self.fake_A.append(self.netG_B(torch.cat([self.real_B, self.fake_A[-1]], 1))) #F1
                self.rec_B.append(self.netG_A(torch.cat([self.fake_A[-1], self.rec_B[-1]], 1)))  #G2


            # self.optimizer_fb = torch.optim.Adam(
            #     (self.fake_B[-1].detach(), self.fake_A[-1].detach(), self.rec_B[-1].detach(), self.rec_A[-1].detach()),
            #     lr=self.lr, betas=(self.beta1, 0.999))

        else:
            if iter == 0:
                self.fake_B = self.netG_A(torch.cat([self.real_A, fake_B_], 1))
                self.rec_A = self.netG_B(torch.cat([self.fake_B, self.real_A], 1))
                self.fake_A = self.netG_B(torch.cat([self.real_B, fake_A_], 1))
                self.rec_B = self.netG_A(torch.cat([self.fake_A, self.real_B], 1))
            else:
                '''F2*G1,G2*F1; equivalent to G2*F1,F2*G1'''
                self.fake_B = self.netG_A(torch.cat([self.real_A, self.fake_B], 1)) #G1
                self.rec_A = self.netG_B(torch.cat([self.fake_B, self.rec_A], 1)) #F2
                self.fake_A = self.netG_B(torch.cat([self.real_B, self.fake_A], 1)) #F1
                self.rec_B = self.netG_A(torch.cat([self.fake_A, self.rec_B], 1)) #G2
                '''G1,G2,F1,F2;'''
            # _ = self.fake_B_pool.query(self.fake_B)
            # _ = self.fake_A_pool.query(self.fake_A)
            del _
                # self.fake_B = self.netG_A(torch.cat([self.real_A, self.fake_B], 1))
                # self.rec_B = self.netG_A(torch.cat([self.fake_A, self.rec_B], 1))
                # self.rec_A = self.netG_B(torch.cat([self.fake_B, self.rec_A], 1))
                # self.fake_A = self.netG_B(torch.cat([self.real_B, self.fake_A], 1))
        # self.fake_A[-1]
        # self.rec_B[-1]
        # self.rec_A[-1]
        # print(self.fake_B[-1].requires_grad)


        # self.fake_B[-1] =
        # self.fake_A[-1] =
        # self.rec_A[-1] =
        # self.rec_B[-1] =
    def backward_D_basic(self, netD, real, fake, ):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # if iter == 0:
        #     self.pred_real[name].append(netD(torch.cat([real, torch.ones_like(real[:,0].unsqueeze(1))],1)))
        # else:
        #     self.pred_real[name].append(netD(torch.cat([real, nn.functional.interpolate(self.pred_real[name][-1], (real.shape[2],real.shape[3]), mode='bilinear')], 1)))

        # pred_fake = netD(torch.cat([fake.detach(), torch.zeros_like(real[:,0].unsqueeze(1))], 1))
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()#torch.ones(1,1,30,30).to(self.device), retain_graph=False)
        return loss_D

    def backward_D_A(self,iter):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B[-1])
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B,)

    def backward_D_B(self,iter):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A[-1])
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A,)
    # def backward_D(self, iter):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake = self.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Real
    #     real_AB = torch.cat((self.real_A, self.real_B), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()

    def backward_G(self, iter):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(torch.cat([self.real_B, self.real_B], 1))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(torch.cat([self.real_A, self.real_A], 1))
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # if iter == 0:
        #     # print(self.fake_B[-1].shape)
        #     # print(self.fake_B[-1][:,1,:,:].unsqueeze(1).shape)
        #     # sys.exit()
        #     pred_fake_A = self.netD_A(torch.cat([self.fake_B[-1], torch.zeros_like(self.fake_B[-1][:,0].unsqueeze(1))], 1))
        #     pred_fake_B = self.netD_B(torch.cat([self.fake_A[-1], torch.zeros_like(self.fake_A[-1][:,0].unsqueeze(1))], 1))
        # else:
        #     pred_fake_A = self.netD_A(torch.cat([self.fake_B[-1], nn.functional.interpolate(self.fb_map_A[-1], (self.fake_B[-1].shape[2],self.fake_B[-1].shape[3]), mode='bilinear')], 1))
        #     pred_fake_B = self.netD_B(torch.cat([self.fake_A[-1], nn.functional.interpolate(self.fb_map_B[-1], (self.fake_B[-1].shape[2],self.fake_B[-1].shape[3]), mode='bilinear')], 1))
        # self.fb_map_A.append(pred_fake_A)
        # self.fb_map_B.append(pred_fake_B)

        # GAN loss D_A(G_A(A))
        self.D_A_maps.append(self.netD_A(self.fake_B[-1]))
        # print('DA shape {}'.format(self.D_A_maps[-1].shape))
        # self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B[-1]), True)
        self.loss_G_A = self.criterionGAN(self.D_A_maps[-1], True)

        # print(self.loss_G_A.shape)

        # sys.exit()
        self.D_B_maps.append(self.netD_B(self.fake_A[-1]))
        # GAN loss D_B(G_B(B))
        # self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A[-1]), True)
        self.loss_G_B = self.criterionGAN(self.D_B_maps[-1], True)

        # del pred_fake_A, pred_fake_B
        # print(real.shape)
        # print(fake.shape)
        # fb_map = nn.functional.interpolate(pred_real, (real.shape[2],real.shape[3]), mode='bilinear')
        # print('map shape {}'.format(fb_map.shape))
        # sys.exit()
        # loss_D_real = self.criterionGAN(pred_real, True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A[-1], self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B[-1], self.real_B) * lambda_B
        # combined loss and calculate gradients
        # if iter >= 1:
            # self.loss_progress = 0.1 * torch.max(torch.zeros(1).to(self.device), 5e-4 + torch.norm(self.D_A_maps[-2]-self.D_A_maps[-1])) + torch.max(torch.zeros(1).to(self.device), 5e-4 + torch.norm(self.D_B_maps[-2]-self.D_B_maps[-1]))
            # print('ploss {}'.format(self.loss_progress))
        self.loss_inter_iter = 0.0
        self.loss_progress = 0.0
        if iter == 1 or iter == 2:
            '''new inter-iter loss'''
            self.loss_inter_iter = lambda_C * (self.inter_iter_loss(self.fake_B[-1], self.fake_B[-2]) +
                                   self.inter_iter_loss(self.fake_A[-1], self.fake_A[-2]) +
                                   self.inter_iter_loss(self.rec_A[-1], self.rec_A[-2]) +
                                   self.inter_iter_loss(self.rec_B[-1], self.rec_B[-2]))
            # self.loss_progress = 8 * torch.max(torch.zeros(1).to(self.device),  torch.mean(self.D_A_maps[-2])-torch.mean(self.D_A_maps[-1])) + torch.max(torch.zeros(1).to(self.device),  torch.mean(self.D_B_maps[-2])-torch.mean(self.D_B_maps[-1]))
            self.loss_progress = 2 * (torch.max(torch.zeros_like(self.D_A_maps[-1]).to(self.device),  (-5e-5) + self.D_A_maps[-2]-self.D_A_maps[-1]).mean() + 1.5*torch.max(torch.zeros_like(self.D_B_maps[-1]).to(self.device), (-5e-5) + self.D_B_maps[-2]-self.D_B_maps[-1]).mean())

            'for logging'
            self.loss_progress_ = self.loss_progress.item()
            self.loss_inter_iter_ = self.loss_inter_iter.item()
            # print('ploss {}'.format(self.loss_progress))
            # print('iiloss {}'.format(self.loss_inter_iter))
            # print('D_A_maps {}'.format(torch.mean(self.D_A_maps[-1])))
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_inter_iter + self.loss_progress
        # self.other_losses = self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_inter_iter + self.loss_progress
        # print('gloss {}'.format(self.loss_G))

        self.loss_G.backward()#torch.ones_like(self.loss_G).to(self.device), retain_graph=True)

        # print('gloss {}'.format(self.loss_G.sum()))

        # print('gloss {}'.format(self.loss_G.mean()))

        # self.other_losses.backward()
        # print('oloss {}'.format(self.other_losses))

        # self.fake_B[-1] = self.fake_B[-1].detach()
        # self.fake_A[-1] = self.fake_A[-1].detach()
        # self.rec_A[-1] = self.rec_A[-1].detach()
        # self.rec_B[-1] = self.rec_B[-1].detach()


    def optimize_parameters(self):
        self.fake_B = []
        self.fake_A = []
        self.rec_A = []
        self.rec_B = []
        self.D_A_maps = []
        self.D_B_maps = []

        #
        # self.fb_map_A = []
        # self.fb_map_B = []
        # self.pred_real = {'A':[],'B':[]}
        # with torch.autograd.set_detect_anomaly(True):
        for iter in range(self.num_iter):
            # print('iter {}'.format(iter))


            self.forward(iter)  # compute fake images: G(A)
            # G_A and G_B

            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs

            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G(iter)  # calculate gradients for D
            # if iter >=1:
            #     print(self.fake_B[-2].requires_grad)
            #     print([self.fake_B[_].grad.sum() for _ in range(iter)])
            self.optimizer_G.step()  # update D's weights
            # self.optimizer_fb.step()

            #self.optimizer_fb.step()
            # D_A and D_B

            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()  # set G's gradients to zero
            self.backward_D_A(iter)  # calculate gradients for D_A
            self.backward_D_B(iter)  # calculate graidents for D_B
            # print(' len {}'.format(len(self.fb_map_A)))
            self.optimizer_D.step()  # udpate G's weights

            # print(self.fake_B[-1].is_leaf)
            # print(self.fake_B[-1].requires_grad)
            # print(self.fake_B[-1].grad_fn)
            # if iter >=1:
            #     print(self.fake_B[-2].requires_grad)
            #     print([self.fake_B[_].grad.sum() for _ in range(iter)])

            self.fake_B[-1] = self.fake_B[-1].detach()
            # self.fake_B[-1].requires_grad_()
            # print(self.fake_B[-1].is_leaf)
            # print(self.fake_B[-1].requires_grad)
            # print(self.fake_B[-1].grad_fn)
            # print('\n\n\n')
            # self.fake_B[-1]=self.fake_B[-1].clone()
            # self.fake_B[-1].retain_grad()
            # print(self.fake_B[-1].is_leaf)
            # print(self.fake_B[-1].grad)
            self.fake_A[-1] = self.fake_A[-1].detach()
            self.rec_A[-1] = self.rec_A[-1].detach()
            self.rec_B[-1] = self.rec_B[-1].detach()
            self.D_A_maps[-1] = self.D_A_maps[-1].detach()
            self.D_B_maps[-1] = self.D_B_maps[-1].detach()

                # self.fb_map_A[-1] = self.fb_map_A[-1].detach()
                # self.fb_map_B[-1] = self.fb_map_B[-1].detach()
                # self.pred_real['A'][-1] = self.pred_real['A'][-1].detach()
                # self.pred_real['B'][-1] = self.pred_real['B'][-1].detach()

        # sys.exit()
                # print(len(self.pred_real))

        # sys.exit()
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        if self.isTrain:
            self.fake_B = torch.cat([_ for _ in self.fake_B], 2)
            self.fake_A = torch.cat([_ for _ in self.fake_A], 2)
            self.rec_B = torch.cat([_ for _ in self.rec_B], 2)
            self.rec_A = torch.cat([_ for _ in self.rec_A], 2)

            self.D_A_maps = torch.cat([_ for _ in self.D_A_maps], 2)
            self.D_B_maps = torch.cat([_ for _ in self.D_B_maps], 2)

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
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):

                errors_ret[name] = float(getattr(self, 'loss_' + name)) # float(...) works for both scalar tensor and float number
        return errors_ret