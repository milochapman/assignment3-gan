import torch
from torch import nn


def train_gan(gen, disc, data_loader, z_dim=100, lr=2e-4, betas=(0.5, 0.999), device="cpu", epochs=5):
    gen = gen.to(device)
    disc = disc.to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for real, _ in data_loader:
            real = real.to(device)
            bs = real.size(0)

            noise = torch.randn(bs, z_dim, device=device)
            fake = gen(noise)

            disc_real = disc(real).view(-1)
            disc_fake = disc(fake.detach()).view(-1)
            loss_d = (criterion(disc_real, torch.ones_like(disc_real)) +
                      criterion(disc_fake, torch.zeros_like(disc_fake))) / 2
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            output = disc(fake).view(-1)
            loss_g = criterion(output, torch.ones_like(output))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        print(f"Epoch [{epoch+1}/{epochs}] D: {loss_d.item():.4f} G: {loss_g.item():.4f}")
