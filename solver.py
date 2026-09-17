import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler
from losses_pytorch.dice_loss import GDiceLossV2,SoftDiceLoss
from losses_pytorch.boundary_loss2 import BoundaryLoss
import segmentation_models_pytorch as smp
import tqdm
from torch.optim import lr_scheduler
from ranger import Ranger
import logging
DE=False
class Solver(object):
    # def __init__(self, config, train_valid_loader):
    def __init__(self, config, train_loader, valid_loader):
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s %(levelname)s %(message)s', 
                            datefmt='%Y-%m-%d %H:%M', 
                            handlers=[logging.FileHandler(os.path.join(config.model_path, f'{config.model_type}.log'), 'w', 'utf-8'), ])


        # Data loader
        # self.train_valid_loader = train_valid_loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        # self.weight_CE=torch.FloatTensor([1,1,1,2,1,1,1])
        self.criterion = torch.nn.BCELoss()
        self.nll=torch.nn.NLLLoss(ignore_index=-1)
        # self.criterion = GDiceLossV2()
        # self.criterion = SoftDiceLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.n_splits = config.n_splits
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        #self.model_path = config.model_path
        self.result_path = config.result_path
        self.model_path = config.model_path
        self.mode = config.mode

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

    def dice_coef_cat(self, y_pred, y_true, smooth=1e-7, threshold=0.5):
        #Ian
        tp = (y_true*y_pred).sum(axis=(2,3))
        fp = ((1-y_true)*y_pred).sum(axis=(2,3))
        fn = (y_true*(1-y_pred)).sum(axis=(2,3))

        intersection=tp+smooth
        union = tp + fn*0.48 + fp*0.52+smooth
        dice= (intersection/union).mean(0)
        return dice

    def dice_coef_cat_loss(self, y_pred, y_true):
        '''
        Dice loss to minimize. Pass to model as loss during compile statement
        '''
        #score = torch.pow(1 - self.dice_coef_cat(y_pred, y_true), 0.75)*torch.tensor((1.,1.4,1.,1.2,1.4,1.,1.), device='cuda:0')
        score = torch.pow(1 - self.dice_coef_cat(y_pred, y_true), 0.75)
        return score.mean()
    
    def conti_loss(self, y_pred):
        
        return torch.nn.functional.l1_loss(y_pred[:,:,1:,:],y_pred[:,:,:-1,:])+torch.nn.functional.l1_loss(y_pred[:,:,:,1:],y_pred[:,:,:,:-1])

    def Generalized_Dice_Loss(self, y_pred, y_true, smooth=1e-5):

        # y_pred = y_pred[:, 1:, :, :]
        # y_true = y_true[:, 1:, :, :]

        y_true = y_true.float()
        y_true_sum = y_true.sum(-1).sum(-1)
 
        # class_weights = Variable(1. / (y_true_sum * y_true_sum).clamp(min=smooth), requires_grad=False)
        class_weights = 1. / (y_true_sum * y_true_sum).clamp(min=smooth)

        intersect = ((y_pred * y_true).sum(-1).sum(-1)) * class_weights
        intersect = intersect.sum()

        denominator = (((y_pred + y_true).sum(-1).sum(-1)) * class_weights).sum()
        # print(2. * intersect / denominator)

        return  1 - (2. * intersect / denominator.clamp(min=smooth))


    def build_model(self):
        ENCODER_WEIGHTS = 'ssl' #'imagenet'
        ACTIVATION = None # coulsoftmax2dd be None for logits or '' for multicalss segmentation
        # create segmentation model with pretrained encoder

        self.unet= smp.MAnet(
            encoder_weights=ENCODER_WEIGHTS, 
            in_channels=3,
            classes=7, 
            activation=None,
        )
        
        self.optimizer = Ranger(self.unet.parameters() , lr= self.lr, weight_decay=1e-6)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs, eta_min=1e-7)
        self.unet.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()
        
    def reset_model(self):
        for layer in self.unet.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                      self.lr, [self.beta1, self.beta2])

    def compute_accuracy(self,SR,GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

    def tensor2img(self,x):
        img = (x[:,0,:,:]>x[:,1,:,:]).float()
        img = img*255
        return img


    def train(self):
        """Train encoder, generator and discriminator."""
        lr = self.lr
        best_unet_score = -1.
        best_epoch = 0.
        alpha=0.
        
        for epoch in range(int(self.num_epochs)):

            self.unet.train(True)
            epoch_loss = 0

            acc = 0.    # Accuracy
            SE = 0.        # Sensitivity (Recall)
            SP = 0.        # Specificity
            PC = 0.        # Precision
            F1 = 0.        # F1 Score
            JS = 0.        # Jaccard Similarity
            DC = 0.        # Dice Coefficient
            DC2 = 0.       #PM Dice
            DC3 = 0.       #ESM Dice
            length = 0



            for i, (images, GT, imagename) in tqdm.tqdm(enumerate(self.train_loader)):
                if DE and i>10: break
                images = images.to(self.device)
                GT = GT.to(self.device)

                # SR : Segmentation Result ================BCE==================

                #SR_log = self.unet(images)  #logsoftmax
                SR_log=torch.nn.functional.log_softmax(self.unet(images),1)
                SR = torch.exp(SR_log)

                SR_flat = SR.view(SR.size(0),-1)
                GT_flat = GT.view(GT.size(0),-1)
                bce_loss = self.nll((1-SR)*SR_log, torch.argmax(GT,1))

                sfdc_loss = self.dice_coef_cat_loss(SR,GT) #soft Dice Loss
                #sfdc_loss=0
                criterion2 = BoundaryLoss() #BD

               # bd_loss=criterion2(SR,GT)
                bd_loss=0
                alpha=0
                loss = sum([(1-alpha)*(bce_loss*0.01+0.9*sfdc_loss),
                           alpha*bd_loss, self.conti_loss(SR)*0.05, 
                           torch.nn.functional.l1_loss(SR,GT)*0.05
                            ])

                epoch_loss += loss.item()

                # Backprop + optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                acc_,SE_,SP_,PC_,F1_,JS_,DC_ = get_accuracy(SR,GT)
                acc += acc_
                SE += SE_
                SP += SP_
                PC += PC_
                F1 += F1_
                JS += JS_
                DC += DC_
                DC2 += DC_

                length += images.size(0)

            alpha += 0.002
            print('\033[1;92mBCE Loss= %.4f,Dice Loss= %.4f,BD Loss= %.4f,alpha= %.4f\033[0m'%(bce_loss,sfdc_loss,bd_loss,alpha))

            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            DC2 = DC2/length

            print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f' % (
                    epoch+1, self.num_epochs, \
                    epoch_loss,\
                    acc,SE,SP,PC,F1,JS))
            logging.info('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f' % (
                    epoch+1, self.num_epochs, \
                    epoch_loss,\
                    acc,SE,SP,PC,F1,JS))
            print(f'DICE:{DC}')
            logging.info(f'DICE:{DC}')

            if not os.path.exists(os.path.join(self.result_path,'train_result.csv')):
                f = open(os.path.join(self.result_path,'train_result.csv'), 'a', encoding='utf-8', newline='')
                headers = ['Epoch','Loss','Accuracy','Dice','lr','best_epoch','num_epochs','num_epochs_decay','augmentation_prob','batch_size']
                wr = csv.writer(f)
                wr.writerow(headers)
                wr.writerow([epoch+1, epoch_loss, acc, DC, self.lr,'--' ,self.num_epochs,self.num_epochs_decay,self.augmentation_prob,self.batch_size])
                f.close()
            else:
                f = open(os.path.join(self.result_path,'train_result.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow([epoch+1, epoch_loss, acc, DC, self.lr,'--' ,self.num_epochs,self.num_epochs_decay,self.augmentation_prob,self.batch_size])
                f.close()

            #===================================== Validation ====================================#
            self.unet.train(False)
            self.unet.eval()

            acc = 0.    # Accuracy
            SE = 0.        # Sensitivity (Recall)
            SP = 0.        # Specificity
            PC = 0.     # Precision
            F1 = 0.        # F1 Score
            JS = 0.        # Jaccard Similarity
            DC = 0.        # Dice Coefficient
            DC2 = 0.
            length=0
            dd=0
            for i, (images, GT, imagename) in enumerate(self.valid_loader):
                if DE and i>10: break
                images = images.to(self.device)
                GT = GT.to(self.device)
                SR_log=torch.nn.functional.log_softmax(self.unet(images),1)
                SR = torch.exp(SR_log)
                acc_,SE_,SP_,PC_,F1_,JS_,DC_ = get_accuracy(SR,GT)
                acc += acc_
                SE += SE_
                SP += SP_
                PC += PC_
                F1 += F1_
                JS += JS_
                DC += DC_
                DC2 += DC_
                Pred=torch.argmax(SR,1)
                GT_=torch.argmax(GT,1)
                for i in range(1,6):
                    dd+=(2*((Pred==i) & (GT_==i)).sum()/(1+(Pred==i).sum()+ (GT_==i).sum())).item()

                length += images.size(0)
            dd/=(length*5)
            acc = acc/length
            SE = SE/length
            SP = SP/length
            PC = PC/length
            F1 = F1/length
            JS = JS/length
            DC = DC/length
            DC2 = DC2/length
            unet_score = dd #JS + DC

            print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, Dice: %.4f'%(acc,SE,SP,PC,F1,JS,dd))
            print(f'DICE:{DC}')
            logging.info('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f,  Dice: %.4f'%(acc,SE,SP,PC,F1,JS,dd))
            logging.info(f'DICE:{DC}')
            #print('step lr')
            self.scheduler.step()



                # Save Best U-Net model
            if unet_score > best_unet_score:
                best_unet_score = unet_score
                best_epoch = epoch+1
                best_unet = self.unet.state_dict()
                print('\033[1;91mBest %s model score : %.4f\033[0m'%(self.model_type,best_unet_score))
                # unet_path = os.path.join(self.model_path, '%s-f%d-%d-%.3f.pkl' % (self.model_type, fold, best_epoch, DC))
                unet_path = os.path.join(self.model_path, '%s.pkl' % (self.model_type))
                torch.save(best_unet,unet_path)
#                 Save U-Net model per 10 epoch
            if (epoch+1)%100 == 0:
#                     best_epoch = epoch
                best_unet = self.unet.state_dict()
                print('model epoch : %.4f'%(epoch+1))
                torch.save(best_unet,os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f-%d.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob,epoch+1)))

            if not os.path.exists(os.path.join(self.result_path,'train_result.csv')):
                f = open(os.path.join(self.result_path,'train_result.csv'), 'a', encoding='utf-8', newline='')
                headers = ['Epoch','Loss','Accuracy','Dice','lr','best_epoch','num_epochs','num_epochs_decay','augmentation_prob','batch_size']
                wr = csv.writer(f)
                wr.writerow(headers)
                wr.writerow([epoch+1, epoch_loss, acc, DC, self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob,self.batch_size])
                f.close()
            else:
                f = open(os.path.join(self.result_path,'train_result.csv'), 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow([epoch+1, epoch_loss, acc, DC, self.lr,best_epoch,self.num_epochs,self.num_epochs_decay,self.augmentation_prob,self.batch_size])
                f.close()
