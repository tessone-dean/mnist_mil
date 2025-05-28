import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np

class MNISTBags(Dataset):
    def __init__(self, root, train,
                 mean_bag_size=1000, bag_size_std=50,
                 num_bags=600, seed=0,
                 max_pos_fraction=0.10):          # ← NEW: cap on % of 1‑digits
        """
        max_pos_fraction : upper bound on (# of '1's) / bag_size for positives.
                           0.10  → at most 10 % of the bag can be 1‑digits.
        """
        super().__init__()
        self.ds = MNIST(root, train=train, download=True,
                        transform=transforms.ToTensor())
        g = torch.Generator().manual_seed(seed)

        # pre‑index
        idx_by_digit = {d: (self.ds.targets == d).nonzero(as_tuple=True)[0]
                        for d in range(10)}
        idx_non1     = torch.cat([idx_by_digit[d]
                                  for d in range(10) if d != 1])

        self.bags = []
        for _ in range(num_bags):
            # ---------------- bag length -----------------
            L = max(1, int(torch.normal(mean_bag_size, bag_size_std,
                                         size=(1,), generator=g).item()))
            # ---------------- choose label ---------------
            is_pos = torch.rand(1, generator=g).item() < 0.5
            if is_pos:
                # --- decide how many '1's ----------------
                # at least 1, at most floor(max_pos_fraction * L)
                max_n_pos = max(1, int(max_pos_fraction * L))
                n_pos = torch.randint(1, max_n_pos + 1, (1,), generator=g).item()

                # sample WITHOUT replacement inside each set
                ones   = idx_by_digit[1][torch.randperm(
                          len(idx_by_digit[1]), generator=g)[:n_pos]]
                others = idx_non1[torch.randperm(
                          len(idx_non1), generator=g)[:L - n_pos]]
                idxs   = torch.cat([ones, others]).tolist()
                label  = 1.
            else:
                # negative  → zero '1's
                idxs  = idx_non1[torch.randperm(
                        len(idx_non1), generator=g)[:L]].tolist()
                label = 0.

            # optional shuffle so '1's aren’t always first
            idxs = torch.tensor(idxs)[torch.randperm(L, generator=g)].tolist()
            self.bags.append((idxs, label))

    def __len__(self): return len(self.bags)

    def __getitem__(self, i):
        idxs, y = self.bags[i]
        imgs = torch.stack([self.ds[j][0] for j in idxs])   # (K,1,28,28)
        return imgs, torch.tensor(y, dtype=torch.float32)


def collate_bags(batch):
    #computes the longest bag length Kmax and allocates a tensor of shape (B,Kmax,1,28,28)
    #creates a mask tensor of shape (B,Kmax) to indicate which positions are valid
    #only the valid positions are used for attention
    imgs, ys = zip(*batch)
    lengths = [b.shape[0] for b in imgs]
    Kmax = max(lengths)

    padded = torch.zeros(len(batch), Kmax, 1, 28, 28)      # zero‑image padding
    mask   = torch.zeros(len(batch), Kmax, dtype=torch.bool)

    for i,b in enumerate(imgs):
        k = b.shape[0]
        padded[i,:k] = b
        mask[i,:k]   = True
    return padded, torch.stack(ys), mask      # (B,K,C,H,W), (B,), (B,K)


if __name__ == "__main__":
    train_ds = MNISTBags('mnist', train=True, num_bags=6000)
    print(len(train_ds))
    print(train_ds[0][0].shape)
    print(train_ds[0][1])
    print(train_ds[1][0].shape)
    print(train_ds[1][1])
    print(train_ds[2][0].shape)
    print(train_ds[2][1])
    print(train_ds[3][0].shape)
    print(train_ds[3][1])
    print(train_ds[4][0].shape)
    print(train_ds[4][1])