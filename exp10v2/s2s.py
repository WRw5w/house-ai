import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet

def train_self2self(noisy_tensor, sigma, num_iter=3000, lr=1e-4, p=0.3, device='cuda'):
    """
    Self2Self training loop.
    noisy_tensor: [1, C, H, W]
    p: Bernoulli drop probability (0.3 is common)
    """
    model = UNet(in_channels=noisy_tensor.size(1), base_features=32, p_drop=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss(reduction='none')

    model.train()
    for i in range(num_iter):
        optimizer.zero_grad()

        # Bernoulli mask: 1 with probability 1-p, 0 with probability p
        mask = torch.bernoulli(torch.full_like(noisy_tensor, 1 - p)).to(device)

        # Masked input: dropped pixels are set to 0 (or masked out)
        # We can just multiply by mask
        net_input = noisy_tensor * mask

        out = model(net_input)

        # Loss is calculated only on dropped pixels
        # Because we want the network to predict the dropped pixels from the context
        drop_mask = 1 - mask
        loss = (mse_loss(out, noisy_tensor) * drop_mask).sum() / drop_mask.sum()

        loss.backward()
        optimizer.step()

    # Inference (Ensemble)
    model.eval()
    # For inference, Self2Self enables dropout to get different predictions
    # Wait, S2S uses Monte Carlo dropout at inference time. We need to force dropout layers to be active.
    # A simple way is to keep model.train() or explicitly enable dropout modules.
    model.train()

    num_ensemble = 50
    preds = []
    with torch.no_grad():
        for _ in range(num_ensemble):
            # Also apply Bernoulli mask during inference
            mask = torch.bernoulli(torch.full_like(noisy_tensor, 1 - p)).to(device)
            net_input = noisy_tensor * mask
            out = model(net_input)
            preds.append(out)

    # Average the predictions
    final_out = torch.mean(torch.stack(preds), dim=0)
    return final_out
