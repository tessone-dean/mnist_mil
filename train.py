import torch, torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as bce
from dataloader import MNISTBags, collate_bags
from feature_encoder import ConvEncoder
from mil_classifier import MILClassifierMasked

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- objects --------------------------------------------------------------
enc     = ConvEncoder(out_dim=128).to(device) #encoder to get features from MNIST images
mil     = MILClassifierMasked(feature_dim=128, hidden_dim=64, gated=False).to(device) #MIL classifier to classify bags
params  = list(enc.parameters()) + list(mil.parameters()) #parameters of encoder and classifier to be optimized bc we are training end to end
opt     = optim.Adam(params, lr=1e-3) #Adam optimizer with learning rate 0.001

enc.load_state_dict(torch.load('enc_checkpoint.pt'))
enc.eval()


train_ds = MNISTBags('mnist', train=True, mean_bag_size=10000, bag_size_std=200, num_bags=60, max_pos_fraction=0.01) #train dataset, 'mnist' is the directory where the dataset is stored
val_ds   = MNISTBags('mnist', train=False, mean_bag_size=5000, bag_size_std=200, num_bags=10, max_pos_fraction=0.01) #validation dataset

print("Made bags")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, #batch size 32, shuffle True
                          collate_fn=collate_bags) 

#data loader's default collate function stacks samples along a first dimension with torch.stack
#that requires every sample to have the same shape
#custom collate function is used to handle bags of different sizes

val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False,
                          collate_fn=collate_bags)


def top_n_mine(H, mask, attn_net, N=50, keep_hard_neg=True):
    """
    H      : (B, K, d)   instance features  (requires grad)
    mask   : (B, K)      True on real instances
    attn_net : nn.Module mapping (B,K,d) -> (B,K) raw scores (no softmax)
    N      : number of positives (and optionally negatives) to keep
    keep_hard_neg : if True, also keep bottom‑N

    Returns
    -------
    H_sel   : (B,  N  or 2N, d)
    mask_sel: (B,  N  or 2N)   (all True)
    """
    with torch.no_grad():                       # scoring is lightweight
        scores = attn_net(H).masked_fill(~mask, -1e9)  # (B,K)
        top_idx = scores.topk(N,  dim=1).indices       # (B,N)

        if keep_hard_neg:
            bottom_idx = scores.topk(N, dim=1, largest=False).indices
            idx = torch.cat([top_idx, bottom_idx], dim=1)  # (B,2N)
        else:
            idx = top_idx                                  # (B,N)

    # expand so torch.gather can pick vectors
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, H.size(-1))   # (B,*,d)
    H_sel = torch.gather(H, 1, idx_exp)                      # (B,*,d)

    mask_sel = torch.ones(idx.shape, dtype=torch.bool, device=H.device)
    return H_sel, mask_sel

# --- loop -----------------------------------------------------------------
for epoch in range(50):
    #enc.train()
    mil.train()
    tot_loss = tot_correct = n = 0
    for imgs, ys, mask in train_loader: #imgs: images in a bag + padding, ys: label of the bag, mask: mask of the bag 
        B,K,C,H,W = imgs.shape #B: batch size, K: number of images in a bag, C: channels, H: height, W: width
        imgs, ys, mask = imgs.to(device), ys.to(device), mask.to(device) #move to device
        feats = enc(imgs.view(B*K, C, H, W)).view(B, K, -1) #get features from encoder, -1 tells torch to infer the last dimension (as 128)
        feats, mask = top_n_mine(feats, mask, mil.attn, N=100, keep_hard_neg=True)

        logits, _ = mil(feats, mask)            # raw logits, _ is the attention weights
        loss = bce(logits, ys) #binary cross entropy loss works with logits, not probabilities

        opt.zero_grad() #zero the gradients
        loss.backward() #backpropagate the loss
        opt.step() #update the parameters

        preds = (torch.sigmoid(logits) > 0.5).float() #predictions are made by thresholding the logits at 0.5
        tot_correct += (preds == ys).sum().item()
        tot_loss    += loss.item() * B
        n           += B

    print(f'Epoch {epoch:02d} | train loss {tot_loss/n:.4f} | acc {tot_correct/n:.3f}')

    # quick validation
    enc.eval(); mil.eval()
    with torch.no_grad():
        tot_correct = n = 0
        for imgs, ys, mask in val_loader:
            B,K,C,H,W = imgs.shape
            imgs, ys, mask = imgs.to(device), ys.to(device), mask.to(device)
            feats = enc(imgs.view(B*K, C, H, W)).view(B, K, -1)
            logits, _ = mil(feats, mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            tot_correct += (preds == ys).sum().item()
            n += B
    print(f'           val acc  {tot_correct/n:.3f}')



# --- save model ------------------------------------------------------------
torch.save(mil.state_dict(), 'mil_checkpoint.pt')
torch.save(enc.state_dict(), 'enc_checkpoint.pt')

