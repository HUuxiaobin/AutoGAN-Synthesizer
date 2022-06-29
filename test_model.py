import os
import sys
import numpy as np
import torch
import time
import logging
import glob
import argparse
import torchvision
from PIL import Image
from torch.autograd import Variable
import utils
import pytorch_ssim
import skimage
import time

parser = argparse.ArgumentParser("deraining")
parser.add_argument('--data', type=str, default='./paired_brats/test/flair/*.png', help='location of the data corpus')
parser.add_argument('--patch_size', type=int, default=64, help='patch size')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--model_path', type=str, default='eval-EXP_time_gan_loss_v2_perception_multi_fliar_t1_t1ce-20210221-013128/best_psnr_weights.pt')
parser.add_argument('--save_dir', type=str, default='./results/gan_loss_v2_perception_time_kspace/', help='gpu device id')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.save = 'test-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def save_image(img, path):
    print('img shape',img.shape)
    img *= 255
    np.clip(img, 0, 255, out=img)
    Image.fromarray(img.astype('uint8'), 'RGB').save(path)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logging.info("args = %s", args)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled=True

    model = torch.load(args.model_path)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    psnr, ssim,mse = infer(args.data, model)
    logging.info('psnr:%6f ssim:%6f mse:%6f', psnr, ssim,mse)

def infer(data_path, model):
    psnr = utils.AvgrageMeter()
    ssim = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    model.eval()
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    with torch.no_grad():
        for step, pt in enumerate(glob.glob(data_path)):
            #print('pt source',pt)
            #rain_image = np.array(Image.open(pt))
            rain_image = (Image.open(pt)).convert('RGB')
#transforms.Compose([
#                            transforms.ToTensor(),
#    ])
            #print('this is ',pt.split('/')[0]+'/'+pt.split('/')[1]+'/'+pt.split('/')[2]+'/'+'T1'+'/'+os.path.basename(pt))
            target_pt='/'.join(pt.split('/')[:9])+'/'+'t2'+'/'+os.path.basename(pt)
            k_space = Image.open('/'.join(pt.split('/')[:9])+'/'+'flair_k_space'+'/'+os.path.basename(pt))
            t1_image = Image.open('/'.join(pt.split('/')[:9])+'/'+'t1'+'/'+os.path.basename(pt)).convert('RGB')
            t1ce_image = Image.open('/'.join(pt.split('/')[:9])+'/'+'t1ce'+'/'+os.path.basename(pt)).convert('RGB')
            #print('this is t1 space',t1_image.shape)
            #pt.split('/')[0]+'/'+pt.split('/')[1]+'/'+pt.split('/')[2]+'/'+'flair_k_space'+'/'+os.path.basename(pt)
#            clear_image=np.array(Image.open(target_pt))
            clear_image=(Image.open(target_pt)).convert('RGB')
#            print(rain_image.shape,clear_image.shape)
#            clear_image = utils.crop_img(image[:,:image.shape[1]//2,:], base=args.patch_size)
#            rain_image = utils.crop_img(image[:,image.shape[1]//2:,:], base=args.patch_size)

            # # Test on whole image
            input = transforms(rain_image).unsqueeze(dim=0).cuda()
            k_space = transforms(k_space).unsqueeze(dim=0).cuda()
            t1_image = transforms(t1_image).unsqueeze(dim=0).cuda()
            t1ce_image = transforms(t1ce_image).unsqueeze(dim=0).cuda()
            #print('t1_image',t1_image.shape)
            #target = transforms(clear_image).unsqueeze(dim=0).cuda(async=True)
            target = transforms(clear_image).unsqueeze(dim=0).cuda()
            inputs=input
            tik = time.time()
            logits = model(inputs)
            print('inference time',time.time()-tik)
            n = inputs.size(0)

            m=skimage.measure.compare_mse(255*np.clip(logits.detach().cpu().numpy(),0,1), 255*target.detach().cpu().numpy())
            s = pytorch_ssim.ssim(torch.clamp(logits,0,1), target)
            p = utils.compute_psnr(np.clip(logits.detach().cpu().numpy(),0,1), target.detach().cpu().numpy())
            psnr.update(p, n)
            ssim.update(s, n)
            mse.update(m, n)
            #save_image(logits.detach().clone().cpu().numpy().squeeze().transpose(1, 2, 0),os.path.join(args.save_dir, os.path.basename(pt)))
            logging.info('name:%s,psnr:%6f ssim:%6f mse:%6f',os.path.basename(pt), p, s,m)
            print('psnr:%6f ssim:%6f mse:%6f' % (p, s,m))


    return psnr.avg, ssim.avg,mse.avg

if __name__ == '__main__':
  main()