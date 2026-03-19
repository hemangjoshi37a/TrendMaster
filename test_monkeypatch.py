import torch
import torch.nn.modules.linear

if not hasattr(torch.nn.modules.linear, '_LinearWithBias'):
    torch.nn.modules.linear._LinearWithBias = torch.nn.modules.linear.Linear

import __main__
from trendmaster.trendmaster import TransAm, PositionalEncoding

__main__.TransAm = TransAm
__main__.PositionalEncoding = PositionalEncoding

import sys

paths = [
    'c:/Users/nitya/Desktop/18/Training/best_model_multi10.pt',
    'c:/Users/nitya/Desktop/18/Inference/best_model_multi10.pt',
    'c:/Users/nitya/Desktop/18/Inference/best_model.pt',
]

for p in paths:
    try:
        data = torch.load(p, map_location='cpu', weights_only=False)
        print(f"{p}: SUCCESS (Type: {type(data)})")
    except Exception as e:
        print(f"{p}: FAILED ({type(e).__name__}: {e})")
