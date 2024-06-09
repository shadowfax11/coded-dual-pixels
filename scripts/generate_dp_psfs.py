import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys; sys.path.append('.')
import re 
import argparse
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.io import savemat 
import src.render_psf as bwk 
import cv2 
from torchvision import transforms 
from PIL import Image 

parser = argparse.ArgumentParser() 
# data directories 
parser.add_argument('--assets_dir', type=str, default='./assets/')
# optical system parameters
parser.add_argument('--f_number', type=float, default=1.73)
parser.add_argument('--focal_length_mm', type=float, default=4.38) 
parser.add_argument('--in_focus_distance_mm', type=float, default=45.0) 
parser.add_argument('--pixel_pitch_um', type=float, default=2.8) 
parser.add_argument('--max_defocus_blur_size_px', type=float, default=40) 
parser.add_argument('--mask', action='store_true')
parser.add_argument('--mask_filename', type=str, default=None)
# DP PSF parameters
parser.add_argument('--psfs_size', type=int, default=61) 
parser.add_argument('--num_depth_planes', type=int, default=21) 
parser.add_argument('--order', type=int, default=1) 
parser.add_argument('--beta', type=float, default=0.4) 
parser.add_argument('--cutoff', type=float, default=2.5) 
parser.add_argument('--smoothing_strength', type=float, default=7) 
# saving parameters
parser.add_argument('--save_psfs', action='store_true')
parser.add_argument('--psf_save_filename', type=str, default='temp.mat') 
# display parameters
parser.add_argument('--display_psfs', action='store_true') 

def main(): 
    args = parser.parse_args()
    # prepare variables
    f = args.focal_length_mm 
    L = f/args.f_number 
    g = args.in_focus_distance_mm
    p = args.pixel_pitch_um * 1e-3 
    alpha_factor = L*f/(1 - f/g) 
    v = f*g/(g-f) 
    D = args.max_defocus_blur_size_px * p 

    z_vals = L*f*g/(L*f - np.linspace(-1*D, D, args.num_depth_planes)*(g-f))
    print(z_vals) 

    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), 
                                    transforms.RandomVerticalFlip(p=1.0)])
    
    coc_scale = 0.5*L*v/g 
    psfs = np.zeros((2, args.num_depth_planes, args.psfs_size, args.psfs_size))
    for i in range(args.num_depth_planes): 
        z = z_vals[i] 
        coc_size = np.round((coc_scale/p)*(z-g)/z).astype(int) 
        coc_size = 1 if coc_size==0 else coc_size
        if coc_size<0:
            kernel_c, kernel_l, kernel_r = bwk.bw_kernel_generator(2*np.abs(coc_size)+1, 
                                            args.order, args.cutoff, args.beta, args.smoothing_strength)
        else: 
            kernel_c, kernel_r, kernel_l = bwk.bw_kernel_generator(2*np.abs(coc_size)+1, 
                                            args.order, args.cutoff, args.beta, args.smoothing_strength)
        kernel_l = (kernel_l/np.max(kernel_l))
        kernel_r = (kernel_r/np.max(kernel_r)) 

        if args.mask: 
            coded_aperture_mask = cv2.imread(os.path.join(args.assets_dir, args.mask_filename), 0).astype(np.float64)
            coded_aperture_mask = cv2.resize(coded_aperture_mask, dsize = kernel_l.shape)
            coded_aperture_mask = coded_aperture_mask / np.max(coded_aperture_mask)
            if i>(args.num_depth_planes)//2: 
                coded_aperture_mask = transform(Image.fromarray(coded_aperture_mask))
            kernel_l = kernel_l * coded_aperture_mask 
            kernel_r = kernel_r * coded_aperture_mask 
        
        sz = kernel_c.shape[0]
        kernel_c = np.pad(kernel_c, ((int((args.psfs_size-sz)/2),int((args.psfs_size-sz)/2)),))
        kernel_l = np.pad(kernel_l, ((int((args.psfs_size-sz)/2),int((args.psfs_size-sz)/2)),))
        kernel_r = np.pad(kernel_r, ((int((args.psfs_size-sz)/2),int((args.psfs_size-sz)/2)),))
        
        psfs[0,i,:,:] = kernel_l   
        psfs[1,i,:,:] = kernel_r

        # Centre-of-mass calculation for left DP PSF
        x_com_l = 0
        y_com_l = 0
        for i in range(args.psfs_size):
            for j in range(args.psfs_size):
                intensity = kernel_l[i, j]
                x_temp = i*intensity
                x_com_l += x_temp
                y_temp = j*intensity
                y_com_l += y_temp
        y_com_l = y_com_l/(np.sum(kernel_l))
        x_com_l = x_com_l/(np.sum(kernel_l))
        # Centre-of-mass calculation for right DP PSF
        x_com_r = 0
        y_com_r = 0
        for i in range(args.psfs_size):
            for j in range(args.psfs_size):
                intensity = kernel_r[i, j]
                x_temp = i*intensity
                x_com_r += x_temp
                y_temp = j*intensity
                y_com_r += y_temp
        y_com_r = y_com_r/(np.sum(kernel_r))
        x_com_r = x_com_r/(np.sum(kernel_r))
        # Compute difference betwen centre-of-masses for the left, right DP PSFs
        x_diff = round(y_com_r-y_com_l, 2)
        y_diff = round(x_com_r-x_com_l, 2)
        print(f"Depth: {z}\t Distance between Left and Right COMs: x: {x_diff}, y: {y_diff}")
        
        # display PSFs
        if args.display_psfs:
            plt.figure()
            plt.subplot(131); plt.imshow(kernel_c); plt.title("Depth: {:3.2f}mm".format(z))
            plt.subplot(132); plt.imshow(kernel_l); plt.title("Left DP PSF")
            plt.subplot(133); plt.imshow(kernel_r); plt.title("Right DP PSF")
            plt.show(block=False); plt.pause(1); plt.close()
    
    if args.save_psfs: 
        save_dict = {}
        save_dict['PSFstack'] = psfs
        savemat(os.path.join(args.assets_dir, args.psf_save_filename), save_dict)


if __name__=="__main__": 
    main()
