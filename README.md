# PoreBoostGAN
Carbonate rocks are characterized by intricate, multi-scale structures that often present significant challenges to conventional imaging techniques. To address the inherent trade-offs between field of view and resolution, we have developed PoreBoostGAN—a cutting-edge super-resolution model based on the Swin Transformer architecture. This model excels at capturing long-range dependencies and reconstructing high-resolution digital rock models. These models not only closely replicate real physical properties but also allow for the prediction of structures beyond current imaging capabilities. Furthermore, our innovative downsampling strategy ensures the preservation of high-frequency information while maintaining manageable data volumes, thus offering a more detailed and efficient representation of carbonate rocks.
## Preprocessing scripts and super-resolution code associated with this project will be publicly released upon the publication of the related research paper.
## Repository Overview
This repository provides the essentials for utilizing PoreBoostGAN, including:

* 🪐 Dataset Preparation: Workflow for converting 3D digital rock datasets into paired meta files necessary for training and inference.
* ⚡️ Training Configuration: Instructions on how to set up and initiate the training process using customizable configuration files.
* 💥 Inference: Guidelines for applying the pre-trained model to new digital rock datasets.
* 🛸 Extrapolation: Workflow for extending and downsampling digital rock images.
* 🧬 [New] 3D Super-Resolution: Direct support for 3D super-resolution by adjusting input and output channels for enhanced Z-axis reconstruction and improved physical fidelity.
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
Follow the instruction to install the pytorch https://pytorch.org/get-started/locally/
For example:
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
then:
pip install basicsr
```

Dataset Preparation
In the dataset directory, we provide scripts for converting and processing your 3D digital rock data:

`tif2png.py`: Converts 3D TIFF files into 2D PNG slices.

`png2tif.py`: Reconstructs 3D TIFF files from 2D PNG slices.

To prepare your dataset, follow these steps:
```bash
cd data
python tif2png_train_val.py
python extract_subimages.py
python generate_meta_info.py
```
After preprocessing, the 3D digital rock will be separated into slices, and meta information will be generated to facilitate pairing and fast loading. The meta info will be saved in a designated folder for each dataset.
## Training Process
To train the model, simply modify the .yml configuration file to suit your needs:

```bash
python src/train.py -opt options/train/ESRGAN/train_Carbonates_x4_test.yml
python src/train.py -opt options/train/ESRGAN/train_Carbonates_x4_3DSR.yml
```

## A fast 3D Super-resolution and reconstruction workflow
After training, you can quickly perform super-resolution and reconstruct 3D digital rocks using the following steps:
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
```
## 🆕 New Feature: 3D Super-Resolution Support
We have added native support for 3D super-resolution in PoreBoostGAN!
Now, by modifying the number of input and output channels, you can control the amount of context integrated in the Z-direction, enabling more accurate and physically meaningful 3D reconstructions.

Highlights
Flexible Context Control:
The number of input/output channels corresponds to the number of slices considered in the Z direction. Increasing this number incorporates more contextual information along the Z-axis, effectively boosting the reconstruction quality of 3D digital rocks.

Physically Meaningful Z-axis Recovery:
By utilizing more Z-slices as input, the model is able to learn richer spatial correlations and generate super-resolved 3D structures that better match the underlying rock physics.

### How to Use:

In your YAML configuration file `(e.g., options/train/ESRGAN/train_Carbonates_x4_3DSR.yml)`, set input_nc and output_nc to the desired number of channels.
For example, to use 5 slices as input and reconstruct the central slice, set input_nc: 5 and output_nc: 1.
For full 3D super-resolution (e.g., input 3 slices, output 3 slices), set both input_nc and output_nc to 3.

```yaml
# Example in YAML config:
  num_in_ch: n     # Number of input slices (channels)
  num_out_ch: 4n    # Number of output slices (channels)
```

## Citation
If you use PoreBoostGAN in your research, please cite it using the following BibTeX entry:

```bibtex
Meng, Yang; An, Senyou (2024), “PoreBoostGAN”, Mendeley Data, V1, doi: 10.17632/6kvtfb5kts.1
```

## Acknowledgments
We acknowledge the High Performance Computing Center at the Eastern Institute of Technology for supporting the computational requirements of this research. The architecture and development of the super-resolution algorithm are primarily based on the BasicSR framework. We also thank Shenzhen University for scanning a new dataset of biomass carbonate rocks with three resolutions.

## License
The code and model weights are licensed under the MIT license. See `LICENSE` for more details.

## Open Source
https://doi.org/10.5281/zenodo.18809715
