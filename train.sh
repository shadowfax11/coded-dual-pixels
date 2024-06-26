if [ $# -lt 2 ]; then
  echo 1>&2 "$0: not enough arguments"
  exit 2
elif [ $# -gt 2 ]; then
  echo 1>&2 "$0: too many arguments"
  exit 2
fi

CUDA_VISIBLE_DEVICES=$1 python scripts/train.py \
--training_mode train+validate \
--dataroot $2 \
--dataset_type flyingthings3d \
--num_workers 8 --N_renders_override 20000 \
--scene_channels dualpix_mono --light_attenuation_factor 1.00 --seed 42 \
--f_number 1.73 --focal_length_mm 4.38 --in_focus_dist_mm 45 --pixel_pitch_um 2.8 \
--max_defocus_blur_size_px 40 --num_depth_planes 21 --rendering_algorithm ikoma-modified \
--readout_noise_level 0.01 --photon_count_level 5000 --crop_renders_for_valid_convolution \
--analytical_deconv_step None --model_type resunet_ppm --model_output defocus aif \
--channel_dim 64 --normalization bn --final_layer_activation none \
--batch_size 8 --num_epochs 80 --lr_model 0.00003 --scheduler cosine \
--defocus_loss l1 grad --aif_loss l1 grad --wt_defocus_loss 1 --wt_aif_loss 1 \
--depth_metric rmse mae --aif_metric psnr \
--save_freq 10 --print_freq 1600 --expt_savedir ../ --expt_name train_cads \
--psf_data_file blur_kernels_simulated_pixel4_nocode.mat \
--initial_mask_fpath ./assets/coded_aperture_nocode_21x21.png \
--initial_mask_alpha 0 --lr_mask 0.0003 --num_epochs_for_mask_learning 30 \
--add_coded_mask_regularizer --min_desired_light_efficiency 0.5
