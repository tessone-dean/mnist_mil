# Multiple Instance Learning for MNIST Digit Classification

This project implements a Multiple Instance Learning (MIL) approach for classifying bags of MNIST digits. The implementation uses a convolutional neural network for feature extraction and an attention-based MIL classifier for bag-level classification.

## Project Structure

- `train.py`: Main training script for the MIL classifier
- `dataloader.py`: Custom dataset and data loading utilities for MNIST bags
- `feature_encoder.py`: Convolutional neural network for feature extraction
- `mil_classifier.py`: Attention-based MIL classifier implementation
- `visualize_mil.py`: Script for visualizing MIL attention weights
- `mnist_data.py`: MNIST dataset utilities
- `enc_checkpoint.pt`: Pre-trained feature encoder weights
- `mil_checkpoint.pt`: Pre-trained MIL classifier weights

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib (for visualization)

## Model Architecture

The model consists of two main components:

1. **Feature Encoder**: A convolutional neural network that extracts features from individual MNIST digits
   - Output dimension: 128 features per image

2. **MIL Classifier**: An attention-based classifier that processes bags of features
   - Hidden dimension: 64
   - Uses attention mechanism for instance selection
   - Implements hard negative mining for improved training

## Training

The training process:
- Creates bags of MNIST digits with variable sizes
- Uses binary cross-entropy loss
- Implements top-N mining for instance selection
- Supports both training and validation phases
- Saves model checkpoints after training

### Training Parameters
- Learning rate: 1e-3
- Optimizer: Adam
- Batch size: 4
- Training bags: 60 bags with mean size 10000
- Validation bags: 10 bags with mean size 5000
- Maximum positive fraction: 0.01

## Usage

1. Train the model:
```bash
python train.py
```

2. Visualize results:
```bash
python visualize_mil.py
```

## Model Checkpoints

The repository includes pre-trained model checkpoints:
- `enc_checkpoint.pt`: Feature encoder weights
- `mil_checkpoint.pt`: MIL classifier weights

## Visualization

The project includes visualization capabilities for:
- Attention weights
- Instance selection
- Bag-level predictions

Results are saved in `mil_visualizations.pdf`.
