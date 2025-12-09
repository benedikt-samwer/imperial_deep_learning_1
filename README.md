# Deep Learning Assessment Part 1

**Imperial College London**

**MSc Applied Computational Science and Engineering (ACSE)**

**Module:** Deep Learning (2024/25)

**Grade:** A

## Project Overview

This repository contains my solution for the first coursework of the Deep Learning module. The objective was to solve an **image imputation (inpainting)** problem for Magnetic Resonance Imaging (MRI) scans of human heads. The goal was to design and train a neural network to recover missing portions of corrupted MRI images.

## Task Description

The assessment involved three main steps:
1.  **Data Generation**: Using a pre-trained Latent Diffusion Model (LDM) to generate a synthetic dataset of realistic MRI brain scans.
2.  **Data Preparation**: Creating a training pipeline that artificially corrupts the generated images to mimic the missing data patterns found in the test set.
3.  **Model Design & Training**: Implementing and training a deep learning architecture to reconstruct the missing parts of the images.

## Methodology

### 1. Data Generation and Preprocessing
*   **Synthetic Data**: Generated **2048** high-quality MRI images (64x64 pixels) using a provided pre-trained Diffusion Model (`ese-invldm`). This served as the ground truth for training.
*   **Corruption Strategy**: Analyzed the provided test set to extract the specific missing-data masks. These masks were then applied to the synthetic training images to create input-target pairs (corrupted image $\rightarrow$ original image).
*   **Data Augmentation**: Used an 80/20 train-validation split to monitor overfitting.

### 2. Model Architecture: Custom U-Net
I implemented a custom **U-Net** architecture from scratch in PyTorch. The U-Net was chosen for its ability to capture both global context and local details, which is crucial for dense prediction tasks like inpainting.

**Key Architecture Features:**
*   **Encoder-Decoder Structure**: Symmetrical contracting and expanding paths.
    *   **Encoder**: 4 blocks of double convolutional layers (3x3 kernels) with MaxPooling, extracting hierarchical features (channels: 64 $\rightarrow$ 128 $\rightarrow$ 256 $\rightarrow$ 512).
    *   **Bottleneck**: High-dimensional feature representation with 1024 channels.
    *   **Decoder**: 4 blocks using Transposed Convolutions for upsampling, recovering spatial resolution.
*   **Skip Connections**: Concatenates feature maps from the encoder to the decoder to preserve fine-grained spatial information lost during downsampling.
*   **Activation Function**: Used **SiLU** (Sigmoid Linear Unit) instead of ReLU for improved gradient flow and performance.
*   **Regularization**:
    *   **Batch Normalization**: Applied after each convolution to stabilize training.
    *   **Dropout (p=0.2)**: Integrated into convolutional blocks to prevent overfitting.

### 3. Training
*   **Loss Function**: Mean Squared Error (MSE) Loss.
*   **Optimizer**: Adam (Learning Rate: 0.001).
*   **Training Regime**: Trained for **150 epochs** on GPU.
*   **Performance**: The model achieved a very low validation loss (~0.0005), demonstrating strong reconstruction capabilities.

## Repository Structure

*   `Assessment.ipynb`: The main Jupyter Notebook containing the full implementation:
    *   Data generation with Diffusion Models.
    *   `TensorDataset` and `DataLoader` creation.
    *   U-Net class definition (`ConvBlock`, `EncBlock`, `DecBlock`).
    *   Training loop with validation.
    *   Visualization of results.
*   `test_set.npy`: The original dataset of 100 corrupted images.
*   `test_set_nogaps.npy`: The final output containing the 100 reconstructed images.

## Results

The model successfully learned to infer missing anatomical structures in the brain scans. Below is a summary of the training performance:
*   **Final Train Loss**: ~0.0008
*   **Final Validation Loss**: ~0.0005

The implemented U-Net demonstrated that convolution-based architectures with skip connections are highly effective for medical image reconstruction tasks where preserving structural integrity is paramount.

