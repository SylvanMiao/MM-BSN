import torch
import torch.nn as nn
from .masks import CentralMaskedConv2d


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        mid_channels = max(in_channels // reduction ,  1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
          nn.ReLU(inplace=True),
          nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
      avg_out = self.mlp(self.avg_pool(x))
      max_out = self.mlp(self.max_pool(x))
      
      return self.sigmoid(avg_out + max_out)
        
class LocalPerceptronBlock(nn.Module):
  """
  k layer's perception field: (2k+1) * (2k+1)
  Uses center-masked convolution to prevent information leakage
  from the center pixel to its neighbors (blind-spot constraint).
  """
  def __init__(self, channels: int):
     super().__init__()
     self.conv = CentralMaskedConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
     self.bn = nn.BatchNorm2d(channels)
     self.act = nn.ReLU(inplace=True)
  def forward(self, x: torch.Tensor)->torch.Tensor:
    return self.act(self.bn(self.conv(x)) + x) # residual
     
class ColorRefinementBlock(nn.Module):
  def __init__(self, channels: int):
     super().__init__()
     self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias= False)
     self.bn = nn.BatchNorm2d(channels)
     self.act = nn.ReLU(inplace=True)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.act(self.bn(self.conv(x)) + x) # alse residual
     

class LCA(nn.Module):
  def __init__(self,
               in_channels: int,
               out_channels: int = None,
               k: int = 2,
               n_color: int = 2,
               ca_reduction: int = 16,
               ):
    super().__init__()

    out_channels = out_channels or in_channels

    # part 1 : input projection
    self.input_proj = nn.Sequential(
        CentralMaskedConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    # part 2 : local perception (center-masked convs, blind-spot safe)
    assert k >= 1, "k must be >= 1"
    self.local_blocks = nn.Sequential(
        *[LocalPerceptronBlock(out_channels) for _ in range(k)]
    )

    # part 3 : color refinement (1x1 convs, blind-spot safe)
    assert n_color >= 0, "n_color must be >= 0"
    self.color_blocks = nn.Sequential(
        *[ColorRefinementBlock(out_channels) for _ in range(n_color)]
    ) if n_color > 0 else nn.Identity()

    # part 4 : channel attention (global pooling, blind-spot safe)
    self.ca = ChannelAttention(out_channels, reduction=ca_reduction)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      out = self.input_proj(x)
      out = self.local_blocks(out)
      out = self.color_blocks(out)
      out = out * self.ca(out)
      return out


