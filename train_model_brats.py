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
import pytorch_ssim
from search_space import HighResolutionNet
from datasets_gpro import GoProDataset
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from vgg import Vgg16
parser = argparse.ArgumentParser("deraining")
parser.add_argument('--data', type=str, default='./datasets/', help='location of the data')
parser.add_argument('--epochs', type=int, default=2000, help='num of training epochs')
parser.add_argument('--steps', type=int, default=50, help='steps of each epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='init learning rate')
parser.add_argument('--save', type=str, default='EXP_perception_loss', help='experiment name')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--patch_size', type=int, default=64, help='patch size')
parser.add_argument('--gpu', type=str, default='0', help='gpu device ids')
parser.add_argument('--ckt_path', type=str, default='search-EXP-20210208-215941/best_psnr_weights.pt', help='checkpoint path of search')
parser.add_argument("-s","--imagesize",type = int, default = 240)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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


def perception_loss(logits,target):
#------------------perception loss------------------
    #y_c_features=vgg(input)
    y_hat_features = vgg(logits)
    style_features = vgg(target)
    style_gram = [utils.gram(fmap) for fmap in style_features]
            # calculate style loss
    y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
#    style_gram = [utils.gram(fmap) for fmap in style_features]
    style_loss = 0.0
    for j in range(4):
        #style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
        style_loss += MSELoss(y_hat_gram[j], style_gram[j])
    style_loss = STYLE_WEIGHT*style_loss
    aggregate_style_loss += style_loss.data[0]

            # calculate content loss (h_relu_2_2)
#    recon = y_c_features[1]      
#    recon_hat = y_hat_features[1]
    content_loss = CONTENT_WEIGHT*MSELoss(logits, target)
    aggregate_content_loss += content_loss.data[0]

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
    diff_i = torch.sum(torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]))
    tv_loss = TV_WEIGHT*(diff_i + diff_j)
    aggregate_tv_loss += tv_loss.data[0]

            # total loss
    #print('style_loss',style_loss,'content_loss',content_loss,'tv_loss',tv_loss)
    loss = style_loss + content_loss + tv_loss
    return loss

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  logging.info("args = %s", args)

  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.enabled=True

  model = HighResolutionNet(ckt_path=args.ckt_path)
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  model.cuda()

  optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

#  train_samples = Rain800(args.data+'training/', args.steps*args.batch_size, args.patch_size)
#  train_queue = torch.utils.data.DataLoader(train_samples, batch_size=args.batch_size, pin_memory=True)
		

  train_dataset = GoProDataset(
            blur_image_files = './paired_brats/train_flair.txt',
            sharp_image_files = './paired_brats/train_t2.txt',
            root_dir = './paired_brats',
            crop = False,
            crop_size = IMAGE_SIZE,
            rotation=True,
            color_augment=True,
            mirror=True,
            transform = transforms.Compose([
                        transforms.ToTensor(),
#                        transforms.Normalize((0.5,0.5,0.5), (1,1,1))
    ])
)
  train_queue = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)

  test_dataset = GoProDataset(
              blur_image_files = './paired_brats/test_flair.txt',
                sharp_image_files = './paired_brats/test_t2.txt',
                crop=False,
                crop_size=IMAGE_SIZE,
                root_dir = './paired_brats',
                transform = transforms.Compose([
                            transforms.ToTensor(),
#                            transforms.Normalize((0.5,0.5,0.5), (1,1,1))
    ])
)
  valid_queue = DataLoader(test_dataset, batch_size = 1, shuffle=False)

#  val_samples = Rain800(args.data+'test_syn/', 50*args.batch_size, args.patch_size)
#  valid_queue = torch.utils.data.DataLoader(val_samples, batch_size=args.batch_size, pin_memory=True)

  best_psnr = 0
  best_psnr_epoch = 0
  best_ssim = 0
  best_ssim_epoch = 0
  best_loss = float("inf") 
  best_loss_epoch = 0
  dtype = torch.cuda.FloatTensor

  for epoch in range(args.epochs):
    logging.info('epoch %d/%d lr %e', epoch+1, args.epochs, scheduler.get_lr()[0])
    aggregate_style_loss = 0.0
    aggregate_content_loss = 0.0
    aggregate_tv_loss = 0.0
    # training
    train(train_queue, model, optimizer,aggregate_style_loss,aggregate_content_loss,aggregate_tv_loss)
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

    scheduler.step()
    logging.info('psnr:%6f ssim:%6f loss:%6f -- best_psnr:%6f best_ssim:%6f best_loss:%6f', psnr, ssim, loss, best_psnr, best_ssim, best_loss)

  logging.info('BEST_LOSS(epoch):%6f(%d), BEST_PSNR(epoch):%6f(%d), BEST_SSIM(epoch):%6f(%d)', best_loss, best_loss_epoch, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)
  torch.save(model, os.path.join(args.save, 'last_weights.pt'))


def train(train_queue, model, optimizer,aggregate_style_loss,aggregate_content_loss,aggregate_tv_loss):
  dtype = torch.cuda.FloatTensor
  vgg = Vgg16().type(dtype)
  for step, (images) in enumerate(train_queue):
    print('--steps:%d--' % step)
#    print('this is input',input)
#    print('this is output',target)
    model.train()
#    print('train_queue',train_queue)
#    target=Variable(images['sharp_image'] - 0.5).cuda()
#    input=Variable(images['blur_image'] - 0.5).cuda()

    target=Variable(images['sharp_image']).cuda()
    input=Variable(images['blur_image']).cuda()
    #print('input',input.shape)

#    input = Variable(train_queue[input]).cuda()
#    target = Variable(train_queue[target]).cuda()
    optimizer.zero_grad()
    logits = model(input)

#------------------perception loss------------------
    #y_c_features=vgg(input)
    y_hat_features = vgg(logits)
    style_features = vgg(target)
    style_gram = [utils.gram(fmap) for fmap in style_features]
            # calculate style loss
    y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
#    style_gram = [utils.gram(fmap) for fmap in style_features]
    style_loss = 0.0
    for j in range(4):
        #style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
        style_loss += MSELoss(y_hat_gram[j], style_gram[j])
    style_loss = STYLE_WEIGHT*style_loss
    aggregate_style_loss += style_loss.data[0]

            # calculate content loss (h_relu_2_2)
#    recon = y_c_features[1]      
#    recon_hat = y_hat_features[1]
    content_loss = CONTENT_WEIGHT*MSELoss(logits, target)
    aggregate_content_loss += content_loss.data[0]

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
    diff_i = torch.sum(torch.abs(logits[:, :, :, 1:] - logits[:, :, :, :-1]))
    diff_j = torch.sum(torch.abs(logits[:, :, 1:, :] - logits[:, :, :-1, :]))
    tv_loss = TV_WEIGHT*(diff_i + diff_j)
    aggregate_tv_loss += tv_loss.data[0]

            # total loss
    #print('style_loss',style_loss,'content_loss',content_loss,'tv_loss',tv_loss)
    loss = style_loss + content_loss + tv_loss
##--------------------------------

###-------------mse loss
#    loss = MSELoss(logits, target)
##---------------------

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

def infer(valid_queue, model):
  psnr = utils.AvgrageMeter()
  ssim = utils.AvgrageMeter()
  loss = utils.AvgrageMeter()

  model.eval()
  with torch.no_grad():
    for _, (images) in enumerate(valid_queue):
      target=Variable(images['sharp_image']).cuda()

      #input=Variable(images['blur_image'] - 0.5).cuda()
      input=Variable(images['blur_image']).cuda()

#      print('this is input',input)
#      input = input.cuda()
#      target = target.cuda()
#      logits = model(input)+0.5
      logits = model(input)
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
