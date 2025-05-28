import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from torch.utils.data import DataLoader
from dataloader import MNISTBags, collate_bags
from feature_encoder import ConvEncoder
from mil_classifier import MILClassifierMasked

def load_model(model_path, encoder_path, device):
    # Initialize models
    encoder = ConvEncoder(out_dim=128).to(device)
    mil = MILClassifierMasked(feature_dim=128, hidden_dim=64, gated=False).to(device)
    
    # Load state dicts
    encoder.load_state_dict(torch.load(encoder_path))
    mil.load_state_dict(torch.load(model_path))
    
    encoder.eval()
    mil.eval()
    return encoder, mil

def visualize_bag(images, attention_scores, pred, true_label, ax):
    """
    Visualize a bag of images with their attention scores
    images: (K, 1, 28, 28) tensor of images
    attention_scores: (K,) tensor of attention scores
    pred: predicted label (0 or 1)
    true_label: true label (0 or 1)
    ax: matplotlib axis to plot on
    """
    K = len(images)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(K)))
    
    # Create a grid of subplots with extra space at the top
    gs = ax.get_gridspec()
    top_ax = ax.figure.add_subplot(gs[0])
    
    # Create a 2D grid of subplots
    img_grid = gs[1].subgridspec(grid_size, grid_size)
    img_axes = []
    for i in range(grid_size):
        for j in range(grid_size):
            img_axes.append(ax.figure.add_subplot(img_grid[i, j]))
    
    # Normalize attention scores for visualization
    attention_scores = attention_scores / attention_scores.sum()
    
    # Sort images and attention scores by attention score (descending)
    sorted_indices = torch.argsort(attention_scores, descending=True)
    sorted_images = images[sorted_indices]
    sorted_scores = attention_scores[sorted_indices]
    
    # Plot images with attention scores
    for i, (img, score) in enumerate(zip(sorted_images, sorted_scores)):
        if i < len(img_axes):  # Only plot if we have space
            img_axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
            img_axes[i].set_title(f'{score:.3f}', fontsize=8)
            img_axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(sorted_images), len(img_axes)):
        img_axes[i].axis('off')
    
    # Add overall classification result at the top
    correct = (pred == true_label)
    color = 'green' if correct else 'red'
    top_ax.text(0.5, 0.5, 
                f'Prediction: {pred:.0f}\nTrue Label: {true_label:.0f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=top_ax.transAxes,
                color=color,
                fontsize=12)
    top_ax.axis('off')
    
    # Adjust layout to prevent overlap
    ax.figure.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    encoder, mil = load_model('mil_checkpoint.pth', 'enc_checkpoint.pth', device)
    
    # Load validation dataset
    val_ds = MNISTBags('mnist', train=False, mean_bag_size=100, bag_size_std=2, num_bags=10)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=collate_bags)
    
    # Create PDF for visualizations
    with PdfPages('mil_visualizations.pdf') as pdf:
        # Visualize a few bags
        num_bags_to_visualize = 10
        for i, (imgs, ys, mask) in enumerate(val_loader):
            if i >= num_bags_to_visualize:
                break
                
            imgs, ys, mask = imgs.to(device), ys.to(device), mask.to(device)
            
            # Get features and predictions
            with torch.no_grad():
                feats = encoder(imgs.view(-1, 1, 28, 28)).view(1, -1, 128)
                logits, attention = mil(feats, mask)
                pred = (torch.sigmoid(logits) > 0.5).float()
            
            # Remove padding
            valid_imgs = imgs[0, mask[0]]
            valid_attention = attention[0, mask[0]]
            
            # Create figure for this bag
            K = len(valid_imgs)
            grid_size = int(np.ceil(np.sqrt(K)))
            fig = plt.figure(figsize=(2*grid_size, 2*grid_size + 1))  # Adjusted size for grid layout
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 4])  # Top row for title, bottom for images
            ax = fig.add_subplot(gs[1, 0])  # Dummy axis for the grid
            
            # Visualize the bag
            visualize_bag(
                valid_imgs,
                valid_attention,
                pred.item(),
                ys.item(),
                ax
            )
            
            # Save to PDF
            pdf.savefig(fig)
            plt.close()

if __name__ == '__main__':
    main()
