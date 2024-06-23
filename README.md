# coded-dual-pixels: Coded Aperture Dual-Pixel Sensing (CADS)
Official Repository for Ghanekar et al. 2024 "Passive Snapshot Coded Aperture Dual-Pixel RGB-D Imaging" 

## Requirements
This code was tested on python=3.8. 
Required libraries: imageio=2.33.1, opencv=4.5.5, scikit-image=0.19.2, einops=0.8.0, lpips=0.1.4, torch=2.3.1+cu118, torchvision=0.18.1+cu118, torchmetrics=1.4.0

To set up the conda environment required for this project, follow these steps:
 ```sh
 git clone https://github.com/shadowfax11/coded-dual-pixels.git
 cd coded-dual-pixels
 conda env create -f environment.yml
 conda activate cads
 ```

## Usage 
For training, run 
```sh
sh train.sh <GPU_ID> <DATA_DIR_FOR_FLYINGTHINGS3D_DATASET>
```

For testing, run
```sh
sh test.sh <GPU_ID> <DATA_DIR_FOR_FLYINGTHINGS3D_DATASET>
```

## Updates
2024-06-12: Code repo is updated and functioning

## References/Credit
Some of the code functions, formatting is taken from other GitHub repos: 
1. https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel
2. https://github.com/ChenyangLEI/sfp-wild
