# PoreBoostGAN
Carbonate rocks are characterized by intricate, multi-scale structures that often present significant challenges to conventional imaging techniques. To address the inherent trade-offs between field of view and resolution, we have developed PoreBoostGAN—a cutting-edge super-resolution model based on the Swin Transformer architecture. This model excels at capturing long-range dependencies and reconstructing high-resolution digital rock models. These models not only closely replicate real physical properties but also allow for the prediction of structures beyond current imaging capabilities. Furthermore, our innovative downsampling strategy ensures the preservation of high-frequency information while maintaining manageable data volumes, thus offering a more detailed and efficient representation of carbonate rocks.
## Preprocessing scripts and super-resolution code associated with this dataset will be publicly released upon the publication of the related research paper.
## Repository Overview
This repository provides the essentials for utilizing PoreBoostGAN, including:

* 🪐 Dataset Preparation: Workflow for converting 3D digital rock datasets into paired meta files necessary for training and inference.
* ⚡️ Training Configuration: Instructions on how to set up and initiate the training process using customizable configuration files.
* 💥 Inference: Guidelines for applying the pre-trained model to new digital rock datasets.
* 🛸 Extrapolation and Downsampling: Workflow for extending and downsampling digital rock images.
## Setup Environment
To get started, clone the repository and set up the environment:

```bash
git clone https://github.com/acse-ym722/PoreBoostGAN.git
cd PoreBoostGAN
```
We provide an environment.yml file that can be used to create a Conda environment. If you only intend to run pre-trained models on a CPU, you may exclude the cudatoolkit and pytorch-cuda dependencies from the file. The code has been tested on Ubuntu 20.04 with CUDA 11.8, PyTorch 2.4 (latest), and Python 3.8.
```bash
conda env create -f environment.yml
conda activate pore
```
Dataset Preparation
In the dataset directory, we provide scripts for converting and processing your 3D digital rock data:

tif2png.py: Converts 3D TIFF files into 2D PNG slices.

png2tif.py: Reconstructs 3D TIFF files from 2D PNG slices.

To prepare your dataset, follow these steps:
```bash
cd data
python tif2png_train_val.py
python extract_subimages.py
python generate_meta_info.py
After preprocessing, the 3D digital rock will be separated into slices, and meta information will be generated to facilitate pairing and fast loading. The meta info will be saved in a designated folder for each dataset.
```
## Training Process
To train the model, simply modify the .yml configuration file to suit your needs:

```bash
python src/train.py -opt options/train/ESRGAN/train_Carbonates_x4_model_2.yml
Fast Super-Resolution & 3D Reconstruction Workflow
After training, you can quickly perform super-resolution and reconstruct 3D digital rocks using the following steps:
```

## A fast 3D Super-resolution and reconstruction workflow
```bash
# Slice your low-resolution digital rock into individual images
python tif2png.py

# Apply super-resolution to all images in the XY plane
python src/app.py -opt options/run.yml 

# Reconstruct the 3D digital rock
python png2tif.py
```
For further refinement, you can upscale the Z-direction using ImageJ:
If you want to avoid denoise model, 
Open ImageJ and load the generated TIFF file.
```bash
Navigate to Image > Scale.
Set the scaling factor for the Z-axis and choose an interpolation method (Bilinear or Bicubic).
Save the result as a new TIFF file.
```
## Extrapolation Mode
PoreBoostGAN supports two extrapolation strategies:
Strategy 1:
Set the input path to the low-resolution images.
Run the model:
```bash
python src/app.py -opt options/run.yml
```
Update the configuration file to set the output path from the previous run as the new input path.
Run the model again:

```bash
python src/app.py -opt options/run.yml
```
Strategy 2:
Set the input path to the high-resolution images.
Run the model:
```bash
python src/app.py -opt options/run.yml
Citation
If you use PoreBoostGAN in your research, please cite it using the following BibTeX entry:
```

## bibtex
```bibtex
Meng, Yang; An, Senyou (2024), “PoreBoostGAN”, Mendeley Data, V1, doi: 10.17632/6kvtfb5kts.1
```

## Acknowledgments
We acknowledge the High Performance Computing Center at the Eastern Institute of Technology for supporting the computational requirements of this research. The architecture and development of the super-resolution algorithm are primarily based on the BasicSR framework. We also thank Shenzhen University for scanning a new dataset of biomass carbonate rocks with three resolutions.

## License
The code and model weights are licensed under the CC-BY-NC license. See LICENSE.txt for more details.
