# PoreBoostGAN
Carbonate rocks exhibit complex, multi-scale features that challenge traditional imaging techniques. To overcome the trade-off between field of view and resolution, we developed a Swin Transformer-based approach that captures long-range features. This method reconstructs high-resolution digital rock models that closely match real physical properties and predict structures beyond current imaging limits. Additionally, our downsampling strategy preserves high-frequency information while keeping data volumes manageable, offering a more detailed and efficient representation of carbonate rocks.

This repository contains:

* 🪐 Basic dataset preparation workflow from 3D digital rock to paired meta files.

* ⚡️ Training config file and how to start training.

* 💥 Inference your digital rock from pretrained model.

* 🛸 extrapolating and downsample workflow


## Setup Environment
First, download and set up the repo:
```bash
git clone https://github.com/acse-ym722/PoreBoostGAN.git
cd PoreBoostGAN
```
We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.
with miniconda successfully installed:
code tested on Ubuntu20.04 cuda11.8 pytorch2.4(latest) python3.8
```bash
conda env create -f environment.yml
conda activate pore
```

## Dataset prepare
In `dataset` folder, we provide tif2png.py and png2tif.py, which are designed for dataset preparation and image reconstruction.
```bash
cd data
python tif2png_train_val.py
python extract_subimages.py
python generate_meta_info.py
```
after preprocecss you can seperate the 3D digital rock into slices and create meta info for pairing and fast load.
meta info will be saved into a folder for each dataset

## Training Process
all you need is to modify the .yml file to setup all config file.
```bash
python src/train.py -opt options/train/ESRGAN/train_Carbonates_x4_model_2.yml # path to your yml
```

## a fast SR & 3D reconstruction workflow
```bash
# After training, assume you have a pretrained model weight
# first slice your low resolution digital rock into single images by runing
python tif2png.py
# SR all the images from XY pannel
python src/app.py -opt options/run.yml # # path to your config file
# reconstruct 3D digital rock
python png2tif.py
# Download tif file and use the imageJ upsample z direction only
Open ImageJ and tif file
Image > Scale
set scaling factor for the Z axis and Interpolation method(Bilinear or Bicubic)
save as a new tif file.
```

## extrapolation mode
```bash
# strategy1
set the input path as low resolution image path
python src/app.py -opt options/run.yml
# the second time you need to change the config file refer the output path of previous run as input path
python src/app.py -opt options/run.yml
# strategy2
set the input path to high resolution image path, then
python src/app.py -opt options/run.yml
```

## BibTeX

```bibtex

```


## Acknowledgments
The computing for this research was supported by High Performance Computing Center at the Eastern Institute of Technology. The main structure of the super-resolution algorithm and development are based on Basicsr. Shenzhen University scanned a new dataset with 3 resolutions of biomass carbonate rocks.


## License
The code and model weights are licensed under CC-BY-NC. See [`LICENSE.txt`](LICENSE.txt) for details.
