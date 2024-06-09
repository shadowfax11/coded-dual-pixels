import math 
import time
import numpy as np 
import torch 
import torch.nn.functional as F 

from src.render import add_noise_poisson, add_noise_readout, clip_signal, add_noise_poisson_gauss
from src.loss import get_loss, get_score 

from utils.log_utils import AverageMeter, ProgressMeter
from utils.vis_utils import tensor2im, tensor2im_colored, save_image, markdown_visualizer, markdown_visualizer_test
from matplotlib import cm

def validate(dataloader_valid, epoch, psfs_meas, coded_mask, model_optics, model_deconv, model, writer, args, logger=None): 
    # set up timers and progress meters and other misc. stuff
    batch_time = AverageMeter('Data+Forward Time', ':6.4f')
    progress_meter_lst = []
    progress_meter_lst.append(batch_time)
    coded_mask_regnorm = AverageMeter('Coded Mask Norm', ':.4e')
    progress_meter_lst.append(coded_mask_regnorm)
    coded_mask_le_factor = AverageMeter('Coded Mask Light Efficiency', ':.4e')
    progress_meter_lst.append(coded_mask_le_factor)
    if "defocus" in args.model_output:
        defocus_losses = []
        for i, loss_item in enumerate(args.defocus_loss):
            defocus_losses.append(AverageMeter(loss_item+' Defocus Loss', ':.4e'))
            progress_meter_lst.append(defocus_losses[i])
        depth_metric_scores = []
        for i, metric in enumerate(args.depth_metric):
            depth_metric_scores.append(AverageMeter(metric+' Depth Metric', ':.4e'))
            progress_meter_lst.append(depth_metric_scores[i])
    if "depth" in args.model_output:
        depth_losses = []
        for i, loss_item in enumerate(args.depth_loss):
            depth_losses.append(AverageMeter(loss_item+' Depth Loss', ':.4e'))
            progress_meter_lst.append(depth_losses[i])
        depth_metric_scores = []
        for i, metric in enumerate(args.depth_metric):
            depth_metric_scores.append(AverageMeter(metric+' Depth Metric', ':.4e'))
            progress_meter_lst.append(depth_metric_scores[i])
    if "aif" in args.model_output:
        aif_losses = []
        for i, loss_item in enumerate(args.aif_loss):
            aif_losses.append(AverageMeter(loss_item+' AIF Loss', ':.4e'))
            progress_meter_lst.append(aif_losses[i])
        aif_metric_scores = []
        for i, metric in enumerate(args.aif_metric):
            aif_metric_scores.append(AverageMeter(metric+' AIF Metric', ':.4e'))
            progress_meter_lst.append(aif_metric_scores[i])
    total_losses = AverageMeter('Total Loss', ':.4e')
    progress_meter_lst.append(total_losses)
    progress = ProgressMeter(len(dataloader_valid), progress_meter_lst, 
                            prefix='Test: ')
    
    if args.training_mode=='validation':
        save_image_dir = "{}/validation_{}".format(args.output_dir, args.dataset_type)
        save_as_type = 'png'
    else: 
        save_image_dir = "{}/{}".format(args.output_dir, epoch)
        save_as_type = 'png' if args.save_renders else 'jpg'
    os.makedirs(save_image_dir, exist_ok=True)
    colors = np.asarray([cm.plasma(i) for i in range(256)])
    colors = (colors[::-1,0:3]*255).astype(np.uint8)

    # set up for validation
    coded_mask.eval()
    model_optics.eval()
    model_deconv.eval()
    model.eval()
    if args.crop_renders_for_valid_convolution:
        crop_h = torch.div(psfs_meas.shape[-2], 2, rounding_mode='trunc')
        crop_w = torch.div(psfs_meas.shape[-1], 2, rounding_mode='trunc')
    light_attenuation_factor = args.light_attenuation_factor

    with torch.no_grad():
        end = time.time()
        for i_sample, data_sample in enumerate(dataloader_valid, 0):
            
            # get and prepare scene data
            x_gt = data_sample.cuda()
            scene_gt, depth_gt, defocus_gt = train_utils.get_in_maps(x_gt, args)

            # render images
            coded_psfs_meas, light_efficiency_factor = coded_mask(psfs_meas, training=False, psf_provided=args.finetune)
            # light_efficiency_factor = torch.mean(torch.sum(coded_psfs_meas, dim=(-2,-1), keepdim=True), dim=-3, keepdim=False)
            y_meas = model_optics.forward(scene_gt, depth_gt, coded_psfs_meas)
            if args.account_for_light_loss:
                light_attenuation_factor = torch.mean(light_efficiency_factor)
            # y_meas = add_noise_poisson(y_meas, args.photon_count_level * light_attenuation_factor)
            # y_meas = add_noise_readout(y_meas, args.readout_noise_level / light_attenuation_factor)
            if args.finetune: 
                a = float(torch.empty(1).uniform_(0.007, 0.015))
                b = float(torch.empty(1).uniform_(0.007, 0.015))
                y_meas = add_noise_poisson_gauss(y_meas, y_meas.detach().clone(), a, b)
            else: 
                y_meas = add_noise_poisson_gauss(y_meas, y_meas.detach().clone(), np.sqrt(1/args.photon_count_level), args.readout_noise_level) # TESTING THIS OUT
            y_meas = clip_signal(y_meas)
            if args.crop_renders_for_valid_convolution:
                scene_gt = scene_gt[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
                depth_gt = depth_gt[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
                defocus_gt = defocus_gt[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
                y_meas = y_meas[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
            
            # analytical deconv steps, if any
            if args.analytical_deconv_step=="None":
                x_analytic = model_deconv(y_meas)                   # identity operation, do nothing
            elif args.analytical_deconv_step=="Deconv":
                psfs_for_deconv = psfs_meas.clone().detach()
                x_analytic = model_deconv(y_meas, psfs_for_deconv)
            elif args.analytical_deconv_step=="Wiener":
                y_meas = y_meas.unsqueeze(2)
                x_analytic = model_deconv.forward(y_meas)
                x_analytic = x_analytic.reshape(x_analytic.size(0), -1, x_analytic.size(-2), x_analytic.size(-1))
            else:
                raise NotImplementedError
            
            # model reconstruction pass
            x_model_input = x_analytic
            if args.add_image_channels_after_analytic_step:
                x_model_input = torch.cat((x_model_input, y_meas[:,:,0,...]), dim=1)
            x_est = model(x_model_input)

            # get predicted and GT maps
            pred_dict = train_utils.get_out_maps(x_est, args.model_output, args.scene_channels, 
                                                args.min_depth_mm, args.max_depth_mm, args.max_defocus_blur_size_px*args.pixel_pitch_um/1000)
            gt_dict = {}
            gt_dict['depth'] = depth_gt
            gt_dict['defocus'] = defocus_gt
            if args.scene_channels == "dualpix_rgb": 
                gt_dict['aif'] = torch.cat((scene_gt[:,[-5]], scene_gt[:,[-3]], scene_gt[:,[-1]]), dim=1)
            elif args.scene_channels == "stdpix_rgb": 
                gt_dict['aif'] = torch.cat((scene_gt[:,[-3]], scene_gt[:,[-2]], scene_gt[:,[-1]]), dim=1)
            else:
                gt_dict['aif'] = scene_gt
            
            # compute losses 
            loss, loss_dict = get_loss(pred_dict, gt_dict, args.defocus_loss, args.depth_loss, args.aif_loss, 
                                        defocus_loss_wt=args.wt_defocus_loss, depth_loss_wt=args.wt_depth_loss, aif_loss_wt=args.wt_aif_loss, 
                                        args=args)
            loss_mask = (1e3*F.relu(args.min_desired_light_efficiency - torch.mean(light_efficiency_factor)))
            if args.add_coded_mask_regularizer:
                loss = loss + loss_mask 
            if args.debug:
                if (math.isnan(loss.item()) or math.isinf(loss.item())):
                    import pdb; pdb.set_trace()
            
            # compute metric scores
            if 'depth' not in pred_dict.keys():
                pred_dict['depth'] = defocus_to_depth(pred_dict['defocus'], 
                                                      args.aperture_mm, args.focal_length_mm, args.in_focus_dist_mm)
            score_dict = get_score(pred_dict, gt_dict, args.depth_metric, args.aif_metric)
            if args.training_mode=="validation":
                err_dict = train_utils.get_error_maps(pred_dict, gt_dict)
            
            coded_mask_regnorm.update(loss_mask.item(), 1)
            coded_mask_le_factor.update(torch.mean(light_efficiency_factor).item(), 1)
            if "defocus" in args.model_output:
                for i, loss_item in enumerate(args.defocus_loss):
                    defocus_losses[i].update(loss_dict['defocus_'+loss_item].item(), x_gt.size(0))
                for i, metric in enumerate(args.depth_metric):
                    depth_metric_scores[i].update(score_dict['depth_'+metric].item(), x_gt.size(0))
            if "depth" in args.model_output:
                for i, loss_item in enumerate(args.depth_loss):
                    depth_losses[i].update(loss_dict['depth_'+loss_item].item(), x_gt.size(0))
                for i, metric in enumerate(args.depth_metric):
                    depth_metric_scores[i].update(score_dict['depth_'+metric].item(), x_gt.size(0))
            if "aif" in args.model_output:
                for i, loss_item in enumerate(args.aif_loss):
                    aif_losses[i].update(loss_dict['aif_'+loss_item].item(), x_gt.size(0))
                for i, metric in enumerate(args.aif_metric):
                    aif_metric_scores[i].update(score_dict['aif_'+metric].item(), x_gt.size(0))
            total_losses.update(loss.item(), x_gt.size(0))

            # save GT, images, predictions, error maps
            if args.training_mode=='validation'and (i_sample%100==0):
                pred_keys = []
                for k, v in pred_dict.items():
                    if k=='aif': 
                        pred_keys.append('aif')
                        if v.shape[1]==1: v = v.repeat(1,3,1,1)
                        save_image(tensor2im(v, cent=0., factor=255./1.), 
                                "{}/im{}_pred_{}.{}".format(save_image_dir, i_sample, k, save_as_type))
                    if k=='depth':
                        pred_keys.append('depth')
                        save_image(tensor2im_colored(v, colors, cent=-1*args.min_depth_mm, factor=255./(args.max_depth_mm-args.min_depth_mm)), 
                                "{}/im{}_pred_{}.{}".format(save_image_dir, i_sample, k, save_as_type))
                gt_keys = []
                for k, v in gt_dict.items():
                    if k=='aif':
                        gt_keys.append('aif')
                        if v.shape[1]==1: v = v.repeat(1,3,1,1)
                        save_image(tensor2im(v, cent=0., factor=255./1.), 
                                "{}/im{}_gt_{}.{}".format(save_image_dir, i_sample, k, save_as_type))
                    if k=='depth':
                        gt_keys.append('depth')
                        save_image(tensor2im_colored(v, colors, cent=-1*args.min_depth_mm, factor=255./(args.max_depth_mm-args.min_depth_mm)), 
                                "{}/im{}_gt_{}.{}".format(save_image_dir, i_sample, k, save_as_type))
                meas_keys = []
                if args.num_psf_channels==2:
                    meas_keys.append('dp_left')
                    save_image(tensor2im(y_meas[0,0,:,:].repeat(1,3,1,1), cent=0., factor=255./1.), 
                                    "{}/im{}_meas_{}.{}".format(save_image_dir, i_sample, "dp_left", save_as_type))
                    meas_keys.append('dp_right')
                    save_image(tensor2im(y_meas[0,1,:,:].repeat(1,3,1,1), cent=0., factor=255./1.), 
                                    "{}/im{}_meas_{}.{}".format(save_image_dir, i_sample, "dp_right", save_as_type))
                elif args.num_psf_channels==1:
                    meas_keys.append('dp_left+right')
                    save_image(tensor2im(y_meas[0,0,:,:].repeat(1,3,1,1), cent=0., factor=255./1.), 
                                    "{}/im{}_meas_{}.{}".format(save_image_dir, i_sample, "img", save_as_type))
                elif args.num_psf_channels==3:
                    meas_keys.append('std_rgb')
                    save_image(tensor2im(y_meas[0,:,:,:].repeat(1,1,1,1), cent=0., factor=255./1.), 
                                    "{}/im{}_meas_{}.{}".format(save_image_dir, i_sample, "img", save_as_type))
                elif args.num_psf_channels==6:
                    meas_keys.append('dp_left_rgb')
                    save_image(tensor2im(y_meas[0,0::2,:,:].repeat(1,1,1,1), cent=0., factor=255./1.), 
                                    "{}/im{}_meas_{}.{}".format(save_image_dir, i_sample, "dp_left", save_as_type))
                    meas_keys.append('dp_right_rgb')
                    save_image(tensor2im(y_meas[0,1::2,:,:].repeat(1,1,1,1), cent=0., factor=255./1.), 
                                    "{}/im{}_meas_{}.{}".format(save_image_dir, i_sample, "dp_right", save_as_type))
                else:
                    raise NotImplementedError
                err_keys = []
                if args.training_mode=="validation":
                    for k, v in err_dict.items():
                        if k=='aif':
                            err_keys.append('aif')
                            if v.shape[1]==1: v = v.repeat(1,3,1,1)
                            save_image(tensor2im(v, cent=0., factor=255./1.), 
                                    "{}/im{}_err_{}.{}".format(save_image_dir, i_sample, k, save_as_type))
                        if k=='depth':
                            err_keys.append('depth')
                            save_image(tensor2im_colored(v, colors, cent=0.0, factor=255./(args.max_depth_mm-args.min_depth_mm)), 
                                    "{}/im{}_err_{}.{}".format(save_image_dir, i_sample, k, save_as_type))
            
            if i_sample % args.print_freq == 0:
                progress.display(i_sample, logger=logger)
            
        if args.training_mode=='validation':
            markdown_visualizer_test(save_image_dir, num=i_sample,
                                        pred_keys=pred_keys, gt_keys=gt_keys, 
                                        meas_keys=meas_keys, err_keys=err_keys, 
                                        flag='reduced')
        
        if 'depth' in args.model_output or 'defocus' in args.model_output: 
            for i, metric in enumerate(args.depth_metric): 
                print_str = "Depth {}: {:.4f}".format(metric, depth_metric_scores[i].avg)
                if logger is not None:
                    logger.info(print_str)
        if 'aif' in args.model_output: 
            for i, metric in enumerate(args.aif_metric):
                print_str = "AIF {}: {:.4f}".format(metric, aif_metric_scores[i].avg)
                if logger is not None: 
                    logger.info(print_str)

        if writer is not None and (not args.training_mode=="validation"):
            no_threshold_mask_pattern = torch.sigmoid(coded_mask.alpha_t*coded_mask.aperture_)
            mask_pattern = ((no_threshold_mask_pattern > 0.5).float())[0]
            writer.add_image("Val/Learned Mask", mask_pattern.detach(), epoch)
            save_image((255*(torch.squeeze(mask_pattern).cpu().detach())).numpy().astype(np.uint8), 
                       os.path.join(args.output_dir, 'ckpt_epochs/ckpt_e{}.png'.format(epoch)))
            writer.add_scalar('Val/Coded_Mask_RegNorm', coded_mask_regnorm.avg, epoch)
            writer.add_scalar('Val/Coded_Mask_LightEfficiency', coded_mask_le_factor.avg, epoch)
            if 'defocus' in args.model_output: 
                for i, loss_item in enumerate(args.defocus_loss):
                    writer.add_scalar('Val/'+loss_item+' Defocus Loss', defocus_losses[i].avg, epoch)
                    if logger is not None:
                        logger.info('Validation (Epoch {:03d}): {} Defocus Loss: {:03.6f}\n'.format(epoch, loss_item, defocus_losses[i].avg))
                for i, metric in enumerate(args.depth_metric):
                    writer.add_scalar('Val/'+metric+' Depth Metric', depth_metric_scores[i].avg, epoch)
                    if logger is not None:
                        logger.info('Validation (Epoch {:03d}): {} Depth Metric: {:03.6f}\n'.format(epoch, metric, depth_metric_scores[i].avg))
            if 'depth' in args.model_output: 
                for i, loss_item in enumerate(args.depth_loss):
                    writer.add_scalar('Val/'+loss_item+' Depth Loss', depth_losses[i].avg, epoch)
                    if logger is not None:
                        logger.info('Validation (Epoch {:03d}): {} Depth Loss: {:03.6f}\n'.format(epoch, loss_item, depth_losses[i].avg))
                for i, metric in enumerate(args.depth_metric):
                    writer.add_scalar('Val/'+metric+' Depth Metric', depth_metric_scores[i].avg, epoch)
                    if logger is not None:
                        logger.info('Validation (Epoch {:03d}): {} Depth Metric: {:03.6f}\n'.format(epoch, metric, depth_metric_scores[i].avg))
            if 'aif' in args.model_output:
                for i, loss_item in enumerate(args.aif_loss):
                    writer.add_scalar('Val/'+loss_item+' AIF Loss', aif_losses[i].avg, epoch)
                    if logger is not None:
                        logger.info('Validation (Epoch {:03d}): {} AIF Loss: {:03.6f}\n'.format(epoch, loss_item, aif_losses[i].avg))
                for i, metric in enumerate(args.aif_metric):
                    writer.add_scalar('Val/'+metric+' AIF Metric', aif_metric_scores[i].avg, epoch)
                    if logger is not None:
                        logger.info('Validation (Epoch {:03d}): {} AIF Metric: {:03.6f}\n'.format(epoch, metric, aif_metric_scores[i].avg))
                        logger.flush()
    
    return total_losses.avg, depth_metric_scores[0].avg

def train_one_epoch(dataloader_train, epoch, psfs_meas, coded_mask, model_optics, model_deconv, model, optimizer_mask, optimizer, writer, args, logger=None):    
    # set up timers and progress meters and other misc. stuff
    batch_time = AverageMeter('Forward Time', ':6.6f')
    data_time = AverageMeter('Data Time', ':6.6f')
    progress_meter_lst = [batch_time, data_time]
    coded_mask_regnorm = AverageMeter('Coded Mask Norm', ':.4e')
    progress_meter_lst.append(coded_mask_regnorm)
    coded_mask_le_factor = AverageMeter('Coded Mask Light Efficiency', ':.4e')
    progress_meter_lst.append(coded_mask_le_factor)
    if "defocus" in args.model_output:
        defocus_losses = []
        for i, loss_item in enumerate(args.defocus_loss):
            defocus_losses.append(AverageMeter(loss_item+' Defocus Loss', ':.4e'))
            progress_meter_lst.append(defocus_losses[i])
    if "depth" in args.model_output: 
        depth_losses = []
        for i, loss_item in enumerate(args.depth_loss):
            depth_losses.append(AverageMeter(loss_item+' Depth Loss', ':.4e'))
            progress_meter_lst.append(depth_losses[i])
    if "aif" in args.model_output:
        aif_losses = []
        for i, loss_item in enumerate(args.aif_loss):
            aif_losses.append(AverageMeter(loss_item+' AIF Loss', ':.4e'))
            progress_meter_lst.append(aif_losses[i])
    total_losses = AverageMeter('Total Loss', ':.4e')
    progress_meter_lst.append(total_losses)    
    progress = ProgressMeter(len(dataloader_train), progress_meter_lst, 
                            prefix="Epoch: [{}]".format(epoch))
    data_time_start = time.time()
    step = 0

    # set up for training
    learn_mask = True
    if args.freeze_mask:
        learn_mask = False 
        coded_mask.eval()
    else: 
        coded_mask.train()
    model_optics.train()
    model_deconv.train()
    model.train()

    if args.crop_renders_for_valid_convolution:
        crop_h = torch.div(psfs_meas.shape[-2], 2, rounding_mode='trunc')
        crop_w = torch.div(psfs_meas.shape[-1], 2, rounding_mode='trunc')
    light_attenuation_factor = args.light_attenuation_factor
    
    for i_batch, sample_batched in enumerate(dataloader_train, 0):
        # logging
        data_time.update(time.time() - data_time_start)
        batch_time_start = time.time()

        # set gradients to zero
        if optimizer_mask is not None:
            optimizer_mask.zero_grad()
        optimizer.zero_grad()

        # get and prepare scene data
        x_gt = sample_batched.cuda()
        scene_gt, depth_gt, defocus_gt = train_utils.get_in_maps(x_gt, args)

        # render images
        coded_psfs_meas, light_efficiency_factor = coded_mask(psfs_meas, training=True, psf_provided=args.finetune)
        # light_efficiency_factor = torch.mean(torch.sum(coded_psfs_meas, dim=(-2,-1), keepdim=True),dim=-3,keepdim=False)
        y_meas = model_optics.forward(scene_gt, depth_gt, coded_psfs_meas)
        if args.account_for_light_loss: 
            light_attenuation_factor = torch.mean(light_efficiency_factor)
        # y_meas = add_noise_poisson(y_meas, args.photon_count_level * light_attenuation_factor)
        # y_meas = add_noise_readout(y_meas, args.readout_noise_level / light_attenuation_factor)
        if args.finetune: 
            a = float(torch.empty(1).uniform_(0.007, 0.015))
            b = float(torch.empty(1).uniform_(0.007, 0.015))
            y_meas = add_noise_poisson_gauss(y_meas, y_meas.detach().clone(), a, b)
        else: 
            y_meas = add_noise_poisson_gauss(y_meas, y_meas.detach().clone(), np.sqrt(1/args.photon_count_level), args.readout_noise_level) # TESTING THIS OUT
        y_meas = clip_signal(y_meas)
        if args.crop_renders_for_valid_convolution:
            scene_gt = scene_gt[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
            depth_gt = depth_gt[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
            defocus_gt = defocus_gt[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
            y_meas = y_meas[..., crop_h:-1*crop_h, crop_w:-1*crop_w]
        
        # analytical deconv steps, if any
        if args.analytical_deconv_step=="None":
            x_analytic = model_deconv(y_meas)                   # identity operation, do nothing
        elif args.analytical_deconv_step=="Deconv":
            psfs_for_deconv = psfs_meas.clone().detach()
            x_analytic = model_deconv(y_meas, psfs_for_deconv)
        elif args.analytical_deconv_step=="Wiener":
            y_meas = y_meas.unsqueeze(2)
            x_analytic = model_deconv.forward(y_meas)
            x_analytic = x_analytic.reshape(x_analytic.size(0), -1, x_analytic.size(-2), x_analytic.size(-1))
        else:
            raise NotImplementedError
        
        # model reconstruction pass
        x_model_input = x_analytic
        if args.add_image_channels_after_analytic_step:
            x_model_input = torch.cat((x_model_input, y_meas[:,:,0,...]), dim=1)
        x_est = model(x_model_input)

        # get predicted and GT maps
        pred_dict = train_utils.get_out_maps(x_est, args.model_output, args.scene_channels, 
                                             args.min_depth_mm, args.max_depth_mm, args.max_defocus_blur_size_px*args.pixel_pitch_um/1000)
        gt_dict = {}
        gt_dict['depth'] = depth_gt
        gt_dict['defocus'] = defocus_gt
        if args.scene_channels == "dualpix_rgb": 
            gt_dict['aif'] = torch.cat((scene_gt[:,[-5]], scene_gt[:,[-3]], scene_gt[:,[-1]]), dim=1)
        elif args.scene_channels == "stdpix_rgb": 
            gt_dict['aif'] = torch.cat((scene_gt[:,[-3]], scene_gt[:,[-2]], scene_gt[:,[-1]]), dim=1)
        else:
            gt_dict['aif'] = scene_gt
        
        # compute losses and back-propagate
        loss, loss_dict = get_loss(pred_dict, gt_dict, args.defocus_loss, args.depth_loss, args.aif_loss, 
                                    defocus_loss_wt=args.wt_defocus_loss, depth_loss_wt=args.wt_depth_loss, aif_loss_wt=args.wt_aif_loss, 
                                    args=args)
        loss_mask = (1e3*F.relu(args.min_desired_light_efficiency - torch.mean(light_efficiency_factor)))
        if args.add_coded_mask_regularizer:
            loss = loss + loss_mask 
        if args.debug:
            if (math.isnan(loss.item()) or math.isinf(loss.item())):
                import pdb; pdb.set_trace()
        loss.backward()
        optimizer.step()
        if learn_mask:
            optimizer_mask.step()
            coded_mask.update_alpha(epoch*len(dataloader_train) + i_batch)

        # display progress
        batch_time.update(time.time() - batch_time_start)
        coded_mask_regnorm.update(loss_mask.item(), 1)
        coded_mask_le_factor.update(torch.mean(light_efficiency_factor).item(), 1)
        if "defocus" in args.model_output:
            for i, loss_item in enumerate(args.defocus_loss):
                defocus_losses[i].update(loss_dict['defocus_'+loss_item].item(), x_gt.size(0))
        if "depth" in args.model_output:
            for i, loss_item in enumerate(args.depth_loss):
                depth_losses[i].update(loss_dict['depth_'+loss_item].item(), x_gt.size(0))
        if "aif" in args.model_output:
            for i, loss_item in enumerate(args.aif_loss):
                aif_losses[i].update(loss_dict['aif_'+loss_item].item(), x_gt.size(0))
        total_losses.update(loss.item(), x_gt.size(0))
        if step % args.print_freq == 0:
            progress.display(step, logger=logger)
        step += 1
        data_time_start = time.time()

    # update tensorboard
    if writer is not None:
        writer.add_scalar('Train/Coded_Mask_RegNorm', coded_mask_regnorm.avg, epoch)
        writer.add_scalar('Train/Coded_Mask_LightEfficiency', coded_mask_le_factor.avg, epoch)
        if 'defocus' in args.model_output:
            for i, loss_item in enumerate(args.defocus_loss):
                writer.add_scalar('Train/'+loss_item+' Defocus Loss', defocus_losses[i].avg, epoch)
                if logger is not None:
                    logger.info('Train (Epoch {:03d}): {} Defocus Loss: {:03.6f}\n'.format(epoch, loss_item, defocus_losses[i].avg))
        if 'depth' in args.model_output: 
            for i, loss_item in enumerate(args.depth_loss):
                writer.add_scalar('Train/'+loss_item+' Depth Loss', depth_losses[i].avg, epoch)
                if logger is not None:
                    logger.info('Train (Epoch {:03d}): {} Depth Loss: {:03.6f}\n'.format(epoch, loss_item, depth_losses[i].avg))
        if 'aif' in args.model_output:
            for i, loss_item in enumerate(args.aif_loss):
                writer.add_scalar('Train/'+loss_item+' AIF Loss', aif_losses[i].avg, epoch)
                if logger is not None:
                    logger.info('Train (Epoch {:03d}): {} AIF Loss: {:03.6f}\n'.format(epoch, loss_item, aif_losses[i].avg))
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        if learn_mask:
            writer.add_scalar('Learning Rate for Learnable Mask', optimizer_mask.param_groups[0]['lr'], epoch)
        no_threshold_mask_pattern = torch.sigmoid(coded_mask.alpha_t*coded_mask.aperture_)
        mask_pattern = ((no_threshold_mask_pattern > 0.5).float())[0]
        writer.add_image("Train/Unthresholded Learned Mask", ((no_threshold_mask_pattern.float())[0]).detach(), epoch)
        writer.add_image("Train/Learned Mask", mask_pattern.detach(), epoch)
        if logger is not None: 
            if learn_mask:
                logger.info('Learning Rate for Learnable Mask {:.06f}\n'.format(optimizer_mask.param_groups[0]['lr']))
            logger.info('Learning Rate {:.06f}\n'.format(optimizer.param_groups[0]['lr']))
            logger.flush()
    
    return coded_mask, model_deconv, model