import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import numpy as np

class RBFKernel(nn.Module):
    def __init__(self, sigmas=[1.0, 2.0, 4.0, 8.0, 16.0]):
        super(RBFKernel, self).__init__()
        self.sigmas = torch.tensor(sigmas)

    def forward(self, X, Y):
        # Calcule la distance euclidienne au carré entre les échantillons
        X_expanded = X.unsqueeze(1)
        Y_expanded = Y.unsqueeze(0)
        dist_squared = torch.sum((X_expanded - Y_expanded) ** 2, dim=2)
        
        # Calcule le noyau RBF multi-échelles
        kernel_val = torch.zeros_like(dist_squared)
        for sigma in self.sigmas:
            gamma = 1.0 / (2.0 * sigma ** 2)
            kernel_val += torch.exp(-gamma * dist_squared)
            
        return kernel_val / len(self.sigmas)

class MMDLoss(nn.Module):
    def __init__(self, sigmas=[1.0, 2.0, 4.0, 8.0, 16.0]):
        super(MMDLoss, self).__init__()
        self.kernel = RBFKernel(sigmas)

    def forward(self, X, Y):
        # X: Features Source (Sen1Floods11)
        # Y: Features Target (Phi-Sat-2)
        k_xx = self.kernel(X, X)
        k_yy = self.kernel(Y, Y)
        k_xy = self.kernel(X, Y)
        
        n, m = X.size(0), Y.size(0)
        
        # Équation du Maximum Mean Discrepancy (biaisée ou non biaisée)
        mmd_squared = k_xx.sum() / (n * n) + k_yy.sum() / (m * m) - 2.0 * k_xy.sum() / (n * m)
        return torch.relu(mmd_squared)  # Relu pour éviter les valeurs négatives dues aux approximations numériques

class UnlabeledDomainDataset(Dataset):
    def __init__(self, h5_path, n_samples=None):
        self.h5_path = h5_path
        self.f = h5py.File(h5_path, 'r')
        self.total_samples = self.f['real/images'].shape[0]
        
        if n_samples is None:
            self.indices = np.arange(self.total_samples)
        else:
            self.indices = np.random.choice(self.total_samples, size=n_samples, replace=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # Chargement Source: Sentinel-2B (7 bandes)
        s2b_data = self.f['s2b/images'][real_idx]
        source_tensor = torch.from_numpy(s2b_data).float()
        
        # Chargement Cible: Phi-Sat-2 Réel (8 bandes)
        real_data = self.f['real/images'][real_idx]
        # Alignement strict sur 7 bandes pour correspondre à l'encodeur TerraMind
        target_tensor = torch.from_numpy(real_data[:7, :, :]).float()
        
        return source_tensor, target_tensor

def get_terramind_encoder():
    from terratorch import BACKBONE_REGISTRY
    
    backbone_size = "tiny"
    backbone_name = f"terramind_v1_{backbone_size}"
    
    # Load the backbone directly
    encoder = BACKBONE_REGISTRY.build(
        backbone_name,
        pretrained=True,
        modalities=["S2L1C"],
        bands={"S2L1C": {"B02": 0, "B03": 1, "B04": 2, "B05": 3, "B06": 4, "B07": 5, "B08": 6}}
    )
    return encoder

# Configuration de l'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
h5_path = Path("/shared/projects/phisat2/data/processed/triplets_v1/phisat2_s2b_dataset_v1.h5")

dataset = UnlabeledDomainDataset(h5_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

encoder = get_terramind_encoder().to(device)
encoder.train()  # Dégel du backbone pour ajuster les poids

optimizer = optim.AdamW(encoder.parameters(), lr=1e-5, weight_decay=0.1)
mmd_criterion = MMDLoss().to(device)

epochs = 50

for epoch in range(epochs):
    epoch_loss = 0.0
    
    for batch_idx, (source_imgs, target_imgs) in enumerate(dataloader):
        source_imgs = source_imgs.to(device)
        target_imgs = target_imgs.to(device)
        
        optimizer.zero_grad()
        
        # Extraction des features de l'espace latent (token [CLS] ou Average Pooling)
        # La méthode exacte dépend de la sortie de votre backbone TerraMind
        source_features = encoder(source_imgs)
        target_features = encoder(target_imgs)
        
        # Si le backbone retourne une liste (multi-couches), prendre la dernière
        if isinstance(source_features, list):
            source_features = source_features[-1]
            target_features = target_features[-1]
            
        # Si le backbone retourne un tenseur spatial (B, C, H, W), appliquer un Global Average Pooling
        if len(source_features.shape) == 4:
            source_features = torch.nn.functional.adaptive_avg_pool2d(source_features, (1, 1)).flatten(1)
            target_features = torch.nn.functional.adaptive_avg_pool2d(target_features, (1, 1)).flatten(1)
        elif len(source_features.shape) == 3: # Format séquence ViT (B, N, D)
            # Utilisation du Global Average Pooling sur la séquence (plus robuste si pas de token [CLS])
            source_features = source_features.mean(dim=1)
            target_features = target_features.mean(dim=1)
            
        # Calcul et minimisation de la perte MMD
        loss = mmd_criterion(source_features, target_features)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | MMD Loss: {epoch_loss/len(dataloader):.6f}")

# Save the fine-tuned encoder weights
save_path = Path("/shared/home/elucas/scratch/terra-sat-drift/outputs/terramind_mmd_finetuned.pth")
torch.save(encoder.state_dict(), save_path)
print(f"Fine-tuned encoder weights saved to {save_path}")
