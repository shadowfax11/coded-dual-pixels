import os 
import numpy as np 
from PIL import Image 
import ntpath 
import cv2 
import scipy.signal
import scipy as sp 
import scipy.ndimage 

def save_image(image_numpy, image_path): 
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_16bit_image(image_numpy, image_path): 
    image = cv2.cvtColor(image_numpy, cv2.CV_16U) 
    cv2.imwrite(image_path, image)

def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=255./2.): 
    image_numpy = image_tensor[0].cpu().float().numpy().transpose((1,2,0)) # 1xCxHxW -> HxWxC
    image_numpy = (image_numpy + cent) * factor 
    image_numpy[image_numpy<0] = 0
    if imtype == np.uint16: 
        image_numpy[np.where(image_numpy>2**16-1)] = 2**16-1
    elif imtype == np.uint8: 
        image_numpy[np.where(image_numpy>2**8-1)] = 2**8-1
    else:
        raise ValueError('imtype should be np.uint8 or np.uint16')
    return image_numpy.astype(imtype)

def tensor2im_colormapped(image_tensor, rgb_vals, imtype=np.uint8, cent=1., factor=255./2., mask=None): 
    assert imtype == np.uint8, 'imtype should be np.uint8'
    if len(image_tensor.shape)==4:
        assert image_tensor.shape[1] == 1, 'input tensor should be 1x1xHxW'
    image_numpy = image_tensor[0].cpu().squeeze().float().numpy()   # 1x1xHxW -> HxW
    image_numpy = (image_numpy + cent) * factor
    image_numpy = np.clip(image_numpy, 0, 255)

    image_numpy = rgb_vals[image_numpy.astype(imtype)].astype(imtype)
    if mask is not None: 
        image_numpy[np.where(mask)] = 0 
    return image_numpy

def markdown_visualizer(save_image_dir, num=10,
                        pred_keys=[''], gt_keys=[''], meas_keys=[''], flag="all"):
    f= open("{}/a_visualizer.md".format(save_image_dir), 'w')
    header_str = "|Index|"
    for k in gt_keys:
        header_str += f"GT {k}|" 
    for k in meas_keys:
        header_str += f"Meas {k}|"
    for k in pred_keys:
        header_str += f"Pred {k}|"
    f.writelines(header_str+"\n")

    divider_str = "|:--:|"
    for _ in list(gt_keys)+list(pred_keys)+list(meas_keys):
        divider_str += f":---:|"
    f.writelines(divider_str+"\n")

    if (flag=="reduced"):
        for i in range(0,num+1,10):
            i_str = f"|{i:03d}|"
            for k in gt_keys:
                i_str += f"![in {k}](im{i}_gt_{k}.jpg)|"
            for k in meas_keys:
                i_str += f"![meas {k}](im{i}_meas_{k}.jpg)|"
            for k in pred_keys:
                i_str += f"![pred {k}](im{i}_pred_{k}.jpg)|"
            f.writelines(i_str+"\n")
    else:
        for i in range(num+1):
            i_str = f"|{i:03d}|"
            for k in gt_keys:
                i_str += f"![in {k}](im{i}_gt_{k}.jpg)|"
            for k in meas_keys:
                i_str += f"![meas {k}](im{i}_meas_{k}.jpg)"
            for k in pred_keys:
                i_str += f"![pred {k}](im{i}_pred_{k}.jpg)|"
            f.writelines(i_str+"\n")

def markdown_visualizer_test(save_image_dir, num=10,
                        pred_keys=[''], gt_keys=[''], meas_keys=[''], err_keys=[], flag="all"):
    f= open("{}/a_visualizer.md".format(save_image_dir), 'w')
    header_str = '|<div style="width:50px">Index</div>|'
    for k in gt_keys:
        header_str += f'<div style="width:300px">GT {k}</div>|' 
    for k in meas_keys:
        header_str += f'<div style="width:300px">Meas {k}</div>|'
    for k in pred_keys:
        header_str += f'<div style="width:300px">Pred {k}</div>|'
    for k in err_keys:
        header_str += f'<div style="width:300px">Err {k}</div>|'
    f.writelines(header_str+"\n")

    divider_str = "|:--:|"
    for _ in list(gt_keys)+list(meas_keys)+list(pred_keys)+list(err_keys):
        divider_str += f":---:|"
    f.writelines(divider_str+"\n")

    disp_interval = 1
    if (flag=="reduced"):
        disp_interval = 10
    for i in range(0,num+1,disp_interval):
        i_str = f"|{i:03d}|"
        for k in gt_keys:
            i_str += f"![in {k}](im{i}_gt_{k}.png)|"
        for k in meas_keys:
            i_str += f"![meas {k}](im{i}_meas_{k}.png)|"
        for k in pred_keys:
            i_str += f"![pred {k}](im{i}_pred_{k}.png)|"
        for k in err_keys:
            i_str += f"![err {k}](im{i}_err_{k}.png)|"
        f.writelines(i_str+"\n")
