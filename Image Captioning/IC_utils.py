"""
Created by Mehmet Zahid Genç
"""

import torch
import IC_config

def save_checkpoint(state):
    print("=> Saving checkpoint")
    torch.save(state, IC_config.checkpoint_filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step