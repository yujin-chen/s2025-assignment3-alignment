import torch
import os
from typing import Union, BinaryIO

def save_checkpoint(model, optimizer, iteration, out: Union[str, os.PathLike, BinaryIO]):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }, out)

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO], model, optimizer):

    checkpoint = torch.load(src, map_location=model.device if hasattr(model, "device") else "cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["iteration"]
