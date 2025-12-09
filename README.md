# Deep Learning Assessment Part 1

**Imperial College London**

**MSc Applied Computational Science and Engineering (ACSE)**

**Module:** Deep Learning (2024/25)

**Grade:** A

## Overview

This repository contains the coursework submission for the first assessment of the Deep Learning module. The project focuses on solving an **image imputation problem** for medical imaging.

The objective was to design and train a deep neural network to recover missing portions of MRI (Magnetic Resonance Imaging) scans of human heads. This task simulates real-world scenarios where patient scans may be incomplete or corrupted to reduce scanning times.

## Project Highlights

- **Deep Learning Framework:** PyTorch
- **Architecture:** Custom U-Net with Skip Connections
- **Key Techniques:** Synthetic Data Generation, Image Inpainting, Convolutional Neural Networks (CNNs)
- **Result:** Successfully reconstructed missing anatomical structures in 64x64 MRI scans with high fidelity.

## Technical Implementation

### 1. Data Strategy: Synthetic Ground Truth

A major challenge of this assessment was the lack of labeled training data (i.e., complete images paired with corrupted ones). To overcome this:

- **Generative Modelling:** I utilised a pre-trained generative model to create a large dataset of realistic, synthetic MRI brain images.
- **Data Pipeline:** Implemented a custom corruption function that artificially removed parts of these synthetic images during training. This created infinite `(corrupted_input, complete_target)` pairs, allowing the model to learn robust inpainting features.

### 2. Model Architecture: U-Net

I designed and implemented a **U-Net** architecture from scratch. The U-Net was chosen for its ability to preserve spatial information through skip connections, which is critical for image reconstruction tasks.

**Key Architectural Choices:**

- **Encoder-Decoder Structure:** A contracting path (encoder) to capture context and a symmetric expanding path (decoder) to enable precise localization.
- **Skip Connections:** Concatenated features from the encoder to the decoder to recover fine-grained spatial details lost during downsampling.
- **Building Blocks:**
  - **Convolutions:** 3x3 kernels with padding to maintain spatial dimensions.
  - **Activation:** `SiLU` (Sigmoid Linear Unit) was selected over ReLU for its improved differentiability and performance in deep networks.
  - **Normalization:** `BatchNorm2d` applied to stabilize training and mitigate internal covariate shift.
  - **Regularization:** `Dropout2d` (p=0.2) included in convolutional blocks to prevent overfitting.
- **Upsampling:** Uses `ConvTranspose2d` layers in the decoder to learn the upscaling filters.

### 3. Training Configuration

- **Loss Function:** `MSELoss` (Mean Squared Error) to strictly penalize pixel-level deviations from the ground truth.
- **Optimizer:** `Adam` (lr=0.001) for adaptive learning rate management.
- **Bottleneck:** A deep bottleneck layer (1024 channels) to capture high-level semantic features before reconstruction.

## Project Structure

- **`Assessment.ipynb`**: The main Jupyter notebook containing the complete solution: data generation, model implementation, training loop, and evaluation.
- **`test_set.npy`**: The provided dataset of 100 corrupted MRI images (64x64 pixels).
- **`test_set_nogaps.npy`**: The final model output containing the 100 restored images.
- **`Instructions.md`**: Original coursework instructions.
- **`References.md`**: Citations and resources used.

## Results

The model was evaluated on the provided `test_set.npy`. Visual inspection confirmed that the network successfully inferred plausible anatomical structures in the missing regions, blending them seamlessly with the surrounding tissue.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

## Usage

To examine the code and results:

1. Open `Assessment.ipynb` in Jupyter Notebook or Google Colab.
2. Run the cells to reproduce the data generation, training, and inference steps.

---

*This work was assessed as part of the MSc ACSE curriculum at Imperial College London.*
