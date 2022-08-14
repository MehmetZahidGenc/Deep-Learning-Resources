import torch
import config
from tqdm import tqdm


def train(netD, netG, dataloader, opt_D, opt_G, l1_loss, bce, g_scaler, d_scaler):

    loop = tqdm(dataloader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x
        y = y

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = netG(x)
            D_real = netD(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = netD(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        netD.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_D)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = netD(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_G.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_G)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )