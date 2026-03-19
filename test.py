import torch
import __main__
from trendmaster.trendmaster import TransAm, PositionalEncoding
import traceback

__main__.TransAm = TransAm
__main__.PositionalEncoding = PositionalEncoding

try:
    torch.load('c:/Users/nitya/Desktop/18/Inference/best_model_multi10.pt', map_location='cpu', weights_only=False)
    print("Loaded successfully")
except Exception as e:
    with open('error.txt', 'w') as f:
        f.write(traceback.format_exc())
