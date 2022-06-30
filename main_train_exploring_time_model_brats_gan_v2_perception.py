import os
import sys
import time
import glob
import math
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
from torchvision import transforms, datasets
import utils
from utils import AverageMeter
import pytorch_ssim
from search_space import HighResolutionNet
from datasets_gpro import MRIDataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from vgg import Vgg16
from discriminator import Discriminator2,define_D
import torch.optim as optim

parser = argparse.ArgumentParser("AutoGAN")
parser.add_argument('--data', type=str, default='./datasets/', help='location of the data')
parser.add_argument('--epochs', type=int, default=1000, help='num of training epochs')
parser.add_argument('--steps', type=int, default=54, help='steps of each epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='init learning rate')
parser.add_argument('--save', type=str, default='EXP_time_gan_loss_v2_perception_multi_fliar_t1_t1ce', help='experiment name')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--patch_size', type=int, default=64, help='patch size')
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids')
parser.add_argument('--ckt_path', type=str, default='search-EXP_k_space-20210220-140401/best_psnr_weights.pt', help='checkpoint path of search')
parser.add_argument("-s","--imagesize",type = int, default = 240)
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--adv_lambda', type=float, default=1e-4,help='adversarial loss weight constant')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')
parser.add_argument('--kernel_size', type=int, default=5, help='size of conv kernel')
parser.add_argument('--lr_decay_step', type=int, default=150,help='learning rate decay step')
parser.add_argument('--log_interval', type=int, default=20,help='log interval, iteration (default: 100)')
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
IMAGE_SIZE = args.imagesize

BATCH_SIZE=args.batch_size
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

MSELoss = torch.nn.MSELoss().cuda()


def perception_loss(logits,target,vgg,aggregate_style_loss,aggregate_content_loss,aggregate_tv_loss):
#------------------perception loss------------------
    #y_c_features=vgg(input)
    STYLE_WEIGHT = 1e5
    CONTENT_WEIGHT = 1e0
    TV_WEIGHT = 1e-7
    y_hat_features = vgg(logits)
    style_features = vgg(target)
    style_gram = [utils.gram(fmap) for fmap in style_features]
    # calculate style loss
    y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
    style_loss = 0.0
    for j in range(4):
        style_loss += MSELoss(y_hat_gram[j], style_gram[j])
    style_loss = STYLE_WEIGHT*style_loss
    aggregate_style_loss += style_loss

    content_loss = CONTENT_WEIGHT*MSELoss(logits, target)
    aggregate_content_loss += content_loss

    # calculate total variation regularization (anisotropic version)
    # https://www.wikiwand.com/en/Total_variation_denoising
    diff_i = torch.sum(torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]))
    tv_loss = TV_WEIGHT*(diff_i + diff_j)
    aggregate_tv_loss += tv_loss

    loss = style_loss + content_loss + tv_loss
    return loss


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
  
    logging.info("args = %s", args)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled=True
    
    print(args.ckt_path)
    model = HighResolutionNet(in_channel=3,ckt_path=args.ckt_path)
#     model = torch.load(args.ckt_path)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model.cuda()
#------define netD

    netD = define_D(3, 64, 3, 'instance', True, 'normal', [0])
    netD.cuda()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
    dtype = torch.cuda.FloatTensor
    vgg = Vgg16().type(dtype)

    train_dataset = MRIDataset(
            flair_image_files='paired_brats/train_flair.txt',
            t2_image_files='paired_brats/train_t2.txt',
            root_dir = 'paired_brats',
            crop = False,
            crop_size = IMAGE_SIZE,
            rotation=True,
            color_augment=True,
            mirror=True,
            transform = transforms.Compose([
                        transforms.ToTensor(),
    ])
)
    train_queue = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

    test_dataset = MRIDataset(
              flair_image_files='paired_brats/test_flair.txt',
                t2_image_files='paired_brats/test_t2.txt',
                crop=False,
                crop_size=IMAGE_SIZE,
                root_dir = 'paired_brats',
                transform = transforms.Compose([
                            transforms.ToTensor(),
    ])
)
    valid_queue = DataLoader(test_dataset, batch_size = 1, shuffle=False)
    best_psnr = 0
    best_psnr_epoch = 0
    best_ssim = 0
    best_ssim_epoch = 0
    best_loss = float("inf") 
    best_loss_epoch = 0
    dtype = torch.cuda.FloatTensor

    d_losses = AverageMeter('D loss')
    total_losses = AverageMeter('Total G loss')
    content_losses = AverageMeter('Content loss')
    adv_losses = AverageMeter('Adversarial loss')

    for epoch in range(args.epochs):
        logging.info('epoch %d/%d lr %e', epoch+1, args.epochs, args.learning_rate)
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0
    #----------- training
        train(train_queue, model,netD,vgg, optimizer,d_losses,total_losses,content_losses,adv_losses,epoch,aggregate_style_loss,aggregate_content_loss,aggregate_tv_loss)
    # validation
        psnr, ssim, loss = infer(valid_queue, model)
    
        if psnr > best_psnr and not math.isinf(psnr):
            torch.save(model, os.path.join(args.save, 'best_psnr_weights.pt'))
            best_psnr_epoch = epoch+1
            best_psnr = psnr
        if ssim > best_ssim:
            torch.save(model, os.path.join(args.save, 'best_ssim_weights.pt'))
            best_ssim_epoch = epoch+1
            best_ssim = ssim
        if loss < best_loss:
            torch.save(model, os.path.join(args.save, 'best_loss_weights.pt'))
            best_loss_epoch = epoch+1
            best_loss = loss

#        scheduler.step()
        logging.info('psnr:%6f ssim:%6f loss:%6f -- best_psnr:%6f best_ssim:%6f best_loss:%6f', psnr, ssim, loss, best_psnr, best_ssim, best_loss)

    logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', best_loss, best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)
    torch.save(model, os.path.join(args.save, 'last_weights.pt'))


def train(train_queue, model,netD, vgg,optimizer,d_losses,total_losses,content_losses,adv_losses,epoch,aggregate_style_loss,aggregate_content_loss,aggregate_tv_loss):
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()

    g_optimizer = optim.Adam(model.parameters(), args.lr, 
                             betas=(0.9,0.999), 
                             eps=1e-8, 
                             weight_decay=0)

    d_optimizer = optim.Adam(netD.parameters(), args.lr, 
                             betas=(0.9,0.999), 
                             eps=1e-8, 
                             weight_decay=0)
    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
#    g_scheduler = optim.lr_scheduler.StepLR(
#        g_optimizer, 
#        step_size=args.lr_decay_step,
#        gamma=0.1)
    
    d_scheduler = optim.lr_scheduler.StepLR(
        d_optimizer, 
        step_size=args.lr_decay_step,
        gamma=0.1)
    
    netD.train()

    for step, (images) in enumerate(train_queue):

        model.train()
        for p in netD.parameters():
            p.requires_grad = True

        target=Variable(images['t2_image']).cuda()
        input=Variable(images['flair_image']).cuda()
        t1_image=Variable(images['t1_image']).cuda()
        t1ce_image=Variable(images['t1ce_image']).cuda()
        flair_k_space=Variable(images['flair_k_space']).cuda()
        t1_k_space=Variable(images['t1_k_space']).cuda()
        t1ce_k_space=Variable(images['t1ce_k_space']).cuda()
        inputs=torch.cat([input,t1_image,t1ce_image,flair_k_space,t1_k_space,t1ce_k_space],1)
        #print('input shape',inputs.shape)
#        optimizer.zero_grad()
        logits = model(inputs)

        real_D = netD(target)
        real_label = torch.ones(real_D.shape).cuda()
        fake_D = netD(logits.detach())
        fake_label = torch.zeros(fake_D.shape).cuda()
        
        real_loss = mse_loss(real_D, real_label)
        fake_loss = mse_loss(fake_D, fake_label)
        d_loss = real_loss + fake_loss
        #print('loss',d_loss)
            # update discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        for p in netD.parameters():
            p.requires_grad = False

        images = next(iter(train_queue))

        content_loss=perception_loss(logits, target,vgg,aggregate_style_loss,aggregate_content_loss,aggregate_tv_loss)
##------------------------------
        adv_loss = bce_loss(netD(logits), real_label)
        total_g_loss = content_loss + args.adv_lambda * adv_loss
            # update generator
        g_optimizer.zero_grad()
        total_g_loss.backward()
        g_optimizer.step()
        d_losses.update(d_loss.item())
        total_losses.update(total_g_loss.item())
        content_losses.update(content_loss.item())
        adv_losses.update(adv_loss.item())
        if step % args.log_interval == 0:
            print('Epoch {:d}/{:d} | Iteration {:d}/{:d} | D loss {:.6f} | Total G loss {:.6f} | Content loss {:.6f} | Adversarial loss {:.6f}'.format(
                    epoch, args.epochs, step, args.steps, d_losses.avg, total_losses.avg, content_losses.avg, adv_losses.avg
                ))
            logging.info('Epoch {:d}/{:d} | Iteration {:d}/{:d} | D loss {:.6f} | Total G loss {:.6f} | Content loss {:.6f} | Adversarial loss {:.6f}'.format(
                    epoch, args.epochs, step, args.steps, d_losses.avg, total_losses.avg, content_losses.avg, adv_losses.avg))
    g_scheduler.step()
    d_scheduler.step()

def infer(valid_queue, model):
    psnr = utils.AvgrageMeter()
    ssim = utils.AvgrageMeter()
    loss = utils.AvgrageMeter()

    model.eval()
    with torch.no_grad():
        for _, (images) in enumerate(valid_queue):
            target=Variable(images['t2_image']).cuda()
            flair_k_space=Variable(images['flair_k_space']).cuda()
            t1_k_space=Variable(images['t1_k_space']).cuda()
            t1ce_k_space=Variable(images['t1ce_k_space']).cuda()
            t1_image=Variable(images['t1_image']).cuda()
            t1ce_image=Variable(images['t1ce_image']).cuda()
            input=Variable(images['flair_image']).cuda()

            inputs=torch.cat([input,t1_image,t1ce_image,flair_k_space,t1_k_space,t1ce_k_space],1)
            logits = model(inputs)
            l = MSELoss(logits, target)
            s = pytorch_ssim.ssim(torch.clamp(logits,0,1), target)
            p = utils.compute_psnr(np.clip(logits.detach().cpu().numpy(),0,1), target.detach().cpu().numpy())
            n = input.size(0)
            psnr.update(p, n)
            ssim.update(s, n)
            loss.update(l, n)
  
    return psnr.avg, ssim.avg, loss.avg

if __name__ == '__main__':
  main()
