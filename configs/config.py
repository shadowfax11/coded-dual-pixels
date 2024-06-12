def config_parser(parser): 
    # scene assets 
    parser.add_argument('--assets_dir', type=str, default='./assets/')
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, choices=['flyingthings3d', 'endoslam'])
    parser.add_argument('--N_renders_override', type=int, default=10000, 
                        help='Number of renders to use for training. Default is 10000. If set to -1, all renders will be used.')
    # rendering params 
    parser.add_argument('--scene_channels', type=str, default='dualpix_mono', 
                        choices=['dualpix_mono', 'dualpix_rgb', 'stdpix_green', 'stdpix_rgb'])
    parser.add_argument('--psf_data_file', type=str, required=True)
    parser.add_argument('--initial_mask_fpath', type=str, default=None) 
    parser.add_argument('--light_attenuation_factor', type=float, default=None)
    parser.add_argument('--min_desired_light_efficiency', type=float, default=0.5) 
    parser.add_argument('--f_number', type=float, required=True) 
    parser.add_argument('--focal_length_mm', type=float, required=True) 
    parser.add_argument('--in_focus_dist_mm', type=float, required=True)
    parser.add_argument('--pixel_pitch_um', type=float, required=True) 
    parser.add_argument('--max_defocus_blur_size_px', type=float, required=True)
    parser.add_argument('--num_depth_planes', type=int, default=21) 
    parser.add_argument('--negative_defocus_only', action='store_true')
    parser.add_argument('--positive_defocus_only', action='store_true')
    parser.add_argument('--rendering_algorithm', type=str, default='ikoma-modified', 
                        choices=['naive', 'xin2021', 'ikoma2021', 'ikoma-modified'])
    parser.add_argument('--crop_renders_for_valid_convolution', action='store_true', 
                        help='If set, renders will be cropped to the valid convolution region.')
    parser.add_argument('--photon_count_level', type=float, default=0, 
                        help='Photon count level to use for rendering (to add poisson shot noise). Default is 0.')
    parser.add_argument('--readout_noise_level', type=float, default=0.01, 
                        help='Readout noise level to use for rendering. Default is 0.01.')
    # model params 
    parser.add_argument('--analytical_deconv_step', type=str, default='None', 
                        choices=['None', 'Deconv', 'Wiener'])
    parser.add_argument('--add_image_channels_after_analytic_step', action='store_true')
    parser.add_argument('--model_type', type=str, default='unet', 
                        choices=['unet', 'dpdnet', 'dpdnet_parallel', 'resunet', 'resunet_ppm'])
    parser.add_argument('--model_output', type=str, nargs='+', 
                        default=['defocus', 'aif'], choices=['defocus', 'depth', 'aif'])
    parser.add_argument('--channel_dim', type=int, default=64) 
    parser.add_argument('--normalization', type=str, default='bn', choices=['bn', 'in'])
    parser.add_argument('--final_layer_activation', type=str, default='relu', choices=['relu', 'tanh', 'sigmoid', 'none'])
    # training params 
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--training_mode', type=str, default='train+validate', choices=['train+validate', 'validation'])
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8) 
    parser.add_argument('--num_epochs', type=int, default=10) 
    parser.add_argument('--num_epochs_for_mask_learning', type=int, default=5) 
    parser.add_argument('--initial_mask_alpha', type=float, default=2.5) 
    parser.add_argument('--freeze_mask', action='store_true')
    parser.add_argument('--lr_mask', type=float, default=0)
    parser.add_argument('--lr_model', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cosine', 
                        choices=['cosine', 'cosine-annealing-warm-restart'])
    parser.add_argument('--load_wts_file', type=str, default=None)
    # loss params 
    parser.add_argument('--add_coded_mask_regularizer', action='store_true') 
    parser.add_argument('--defocus_loss', type=str, nargs='+', 
                        choices=['l1','mse','grad','ssim','smoothl1'])
    parser.add_argument('--depth_loss', type=str, nargs='+', 
                        choices=['l1','mse','grad','ssim','ssim_inv'])
    parser.add_argument('--aif_loss', type=str, nargs='+', 
                        choices=['l1','mse','grad','ssim'])
    parser.add_argument('--wt_defocus_loss', type=float, default=0.00)
    parser.add_argument('--wt_depth_loss', type=float, default=0.00)
    parser.add_argument('--wt_aif_loss', type=float, default=0.00)
    # metrics to compute during validation/testing
    parser.add_argument('--depth_metric', type=str, default=['mse'], nargs='+', 
                        choices=['rmse','mse','absrel','mae','delta1','delta2','delta3','ssim','ai1','ai2','sps'])
    parser.add_argument('--aif_metric', type=str, default=['mse'], nargs='+', 
                        choices=['mse','ssim','psnr','lpips'])
    # printing/logging params 
    parser.add_argument('--save_renders', action='store_true', help='Save renders during validation/testing.')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency to save model. Default: 10')
    parser.add_argument('--print_freq', type=int, default=100, help='Frequency to print losses/metrics/info. Default: 100')
    # input/output related params 
    parser.add_argument('--expt_savedir', type=str, default='./')
    parser.add_argument('--expt_name', type=str, required=True)
    # set default for boolean args
    parser.set_defaults(finetune=False) 
    parser.set_defaults(debug=False)
    parser.set_defaults(resume_training=False)
    parser.set_defaults(save_renders=False)
    return parser