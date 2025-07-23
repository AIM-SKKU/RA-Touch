import torch
import torch.nn as nn

from torch import Tensor
from tvl_enc.transformer_utils import CrossAttention


class TAIntegrator(nn.Module):
    def __init__(self, 
                 feature_dim: int = 768,
                 output_dim: int = 4096
    ):
        super(TAIntegrator, self).__init__()
        self.tac2img_crs_attn = CrossAttention(feature_dim, feature_dim, num_heads=8, qkv_bias=True)
        self.tac2txt_crs_attn = CrossAttention(feature_dim, feature_dim, num_heads=8, qkv_bias=True)
        self.fusion_proj = nn.Linear(feature_dim, output_dim)
        self.fusion_norm = nn.LayerNorm(output_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim//4),
            nn.ReLU(),
            nn.Linear(output_dim//4, output_dim),
        )

    def forward(self, 
                retrieved_img: Tensor,
                retrieved_txt: Tensor,
                input_tac: Tensor,
        ) -> Tensor:
            tac2txt = self.tac2txt_crs_attn(input_tac, retrieved_img)
            tac2img = self.tac2img_crs_attn(input_tac, retrieved_txt)
            
            fusion = tac2img + tac2txt
            fusion = self.fusion_proj(fusion)
            fusion_feats = fusion + \
                        self.fusion_mlp(self.fusion_norm(fusion))
                        
            return fusion_feats

if __name__ == '__main__':
    retrieved_txt = torch.randn((1, 10, 768))
    retrieved_img = torch.randn((1, 10, 768))
    input_tac = torch.randn((1, 1, 768))
    input_vis = torch.randn((1, 1, 768))
    
    m = TAIntegrator()
    out = m(retrieved_img, retrieved_txt, input_tac)
    print(out.shape)