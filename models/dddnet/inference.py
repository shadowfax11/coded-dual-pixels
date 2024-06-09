import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from mydataset import *
import model_test 
import scipy.misc
from PIL import Image
import PIL.Image as pil
import skimage
import PIL.Image as pil
from tqdm import tqdm


parser = argparse.ArgumentParser(description="DDD")
parser.add_argument('--batchsize',type = int, default = 8)
parser.add_argument('--gpu',type=int, default=1)
parser.add_argument('--root_dir', type=str, default ="/home/shreyas/datasets/DP_data_pixel_4")
parser.add_argument('--img_list_t', type=str, default ="./data/my_test.txt")
parser.add_argument('--output_file', type=str, default ="test_results/DDDsys/mydayaset")
parser.add_argument('--modelname', type=str, default = "model_dpd", help="model_nyu")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
args = parser.parse_args()


#Hyper Parameters
METHOD = args.modelname
OUT_DIR = args.output_file
GPU = range(args.gpu)
TEST_DIR = args.root_dir

 
if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

print("init data folders - NYU dataset")

dataset = MyDataset(
    img_list = args.img_list_t,
    root_dir = args.root_dir,
    )
dataloader = DataLoader(dataset, batch_size = args.batchsize, shuffle=False, num_workers=args.workers)

mse = nn.MSELoss().cuda()

Estd_stereo = model_test.YRStereonet_3D()
Esti_stereod = model_test.Mydeblur()
Estd_stereo = torch.nn.DataParallel(Estd_stereo, device_ids=GPU)
Esti_stereod = torch.nn.DataParallel(Esti_stereod, device_ids=GPU)
Estd_stereo.cuda()
Esti_stereod.cuda()


Estd_stereo.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/Estd" + ".pkl")), strict=False)
print("ini load Estd " + " success")
Esti_stereod.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/Esti" + ".pkl")), strict=False)
print("ini load Esti " + " success")
  

def infer():
    pscale = 0.0

    print("Infering...")
    Estd_stereo.eval()
    Esti_stereod.eval()

    with torch.no_grad():
    
        psnr = []
        errors_m = []
        for i, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
          
            image_left = Variable(inputs['left'] - pscale, requires_grad=False).cuda()
            image_right = Variable(inputs['right'] - pscale, requires_grad=False).cuda()

            # image_id = inputs['img_id']
            image_suffix = 'png'

            
            est_blurdispt = Estd_stereo(image_left,image_right)
            deblur_imaget, est_mdispt = Esti_stereod(image_left, image_right, est_blurdispt) 

            # psnr.append(torchPSNR(deblur_imaget, gt_image))

            # print(est_mdispt.min(), est_mdispt.max())
            # imshow(est_mdispt[0],"temp1.png")
            # imshow(est_blurdispt[0], "temp2.png")
            # import sys 
            # sys.exit()
            for i in range(deblur_imaget.size(0)):
                    imshow(est_mdispt[i],f"estdisp{i}.png")
                    imshow(est_blurdispt[i], f"mdisp{i}.png")
                    torchvision.utils.save_image(deblur_imaget[i].data + pscale, f"deblur{i}.png")
                # torchvision.utils.save_image(deblur_imaget[i].data + pscale, OUT_DIR + '/' + str(i) + '_o.' + image_suffix)
                # torchvision.utils.save_image(est_mdispt[i].data + pscale, OUT_DIR + '/' + str(i) + '_mdisp.' + image_suffix)
                # torchvision.utils.save_image(est_blurdispt[i].data + pscale, OUT_DIR + '/' + str(i) + '_blurdisp.' + image_suffix)

def imshow(tensor_image,name):
        # Convert tensor to numpy array
        numpy_image = tensor_image.cpu().numpy()
        # print(numpy_image.shape)
        transposed   = numpy_image
        # The tensor usually has shape (C, H, W) so we need to transpose it to (H, W, C) for visualization
        # transposed = numpy_image.transpose(1, 2, 0)
        # print(transposed.shape)
        transposed = (transposed - np.min(transposed)) / (np.max(transposed) - np.min(transposed))

        plt.imshow(transposed)
        plt.savefig(name, bbox_inches='tight', pad_inches=0)
        plt.axis('off')  # Hide axes
        # plt.show()

          
def torchPSNR(tar_img, prd_img):

    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

if __name__ == '__main__':
    infer()

        

        

