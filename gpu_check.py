import torch
import jax
import jax.numpy as jnp
import os
from jax import vmap, random
from einops import rearrange, repeat
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
print(f"running calculations on: {device}")
print(f"jax devices:={jax.devices()}")
W = torch.ones(size=(3,3))
b = torch.ones(size=(3,3))
result = W@b
print(f"result={result}")
# A = repeat(
#             torch.arange(1, 12 + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=2,
#         ).contiguous()
# print(f"A.shape={A.shape}")
# print(f"A={A}")
