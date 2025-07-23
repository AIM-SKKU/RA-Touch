import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional, Tuple
from tvl_enc.transformer_utils import Attention, CrossAttention


class TGRetriever(nn.Module):
    def __init__(self, 
                 feature_dim: int = 768, 
                 output_dim: int = 768,
                 retrieval_method: str = 'tac2txt',
                 embed_path: Optional[str] = None, 
                 device: str = 'cuda',
    ):
        super(TGRetriever, self).__init__()
        
        self.device = device
        self.tac_attns = nn.ModuleList([
            Attention(dim=feature_dim, num_heads=8, qkv_bias=True)
        ])
        self.tac_norms = nn.ModuleList([
            nn.LayerNorm(feature_dim)
        ])
        self.crs_norm = nn.LayerNorm(feature_dim)
        self.vis_norm = nn.LayerNorm(feature_dim)
        self.proj_norm = nn.LayerNorm(output_dim)
        self.vis_attn = Attention(dim=feature_dim, num_heads=8, qkv_bias=True)
        self.crs_attn = CrossAttention(feature_dim, feature_dim, num_heads=8, qkv_bias=True)
        self.query_proj = nn.Linear(feature_dim, output_dim)
        
        if embed_path is not None:
            with open(embed_path, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                self.img_embs = torch.from_numpy(data['img_embs']).to(device)
                self.text_embs = torch.from_numpy(data['text_embs']).to(device)
                self.img_ids = torch.from_numpy(data['img_ids']).to(device)
            
            if retrieval_method.endswith('img'):
                self.external_embs = self.img_embs.type(torch.float32).to(device)
            elif retrieval_method.endswith('txt'):
                self.external_embs = self.text_embs.type(torch.float32).to(device)
            else:
                raise ValueError("Unsupported retrieval method.")

            json_path = embed_path.replace('embeddings', 'annotations').replace('npz', 'json')
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            self.annotations = {item["id"]: item["caption"] for item in annotations.values()}
    
    def retrieve(self, query_embs: Tensor, top_k: int = 5, distance: str = 'cosine') -> Tuple[Tensor]:
        with torch.no_grad():
            sim_mat = torch.matmul(F.normalize(self.external_embs, dim=-1), query_embs.T)
            sort_inds = torch.argsort(sim_mat.squeeze(), descending=True)
        
        sort_inds = sort_inds.cpu()
        retrieve_img_embds = self.img_embs[sort_inds[:top_k]].to(self.device)
        retrieve_txt_embds = self.text_embs[sort_inds[:top_k]].to(self.device)
        return retrieve_img_embds, retrieve_txt_embds
        
    def diversity_loss(self, query_emb: Tensor) -> Tensor:
        sim_matrix = query_emb @ query_emb.T
        batch_size = query_emb.size(0)
        mask = torch.eye(batch_size, device=query_emb.device).bool()
        diversity_loss = sim_matrix.masked_fill(mask, 0).mean()
        return diversity_loss

    def clip_loss(self, query_emb: Tensor, tac_feats: Tensor) -> Tensor:
        labels = torch.arange(query_emb.shape[0], device=query_emb.device, dtype=torch.long)
        affinity_matrix = query_emb @ tac_feats.T
        row_loss = F.cross_entropy(affinity_matrix, labels)
        col_loss = F.cross_entropy(affinity_matrix.T, labels)
        return (row_loss + col_loss) / 2

    def forward(self, 
                vis_feat: Tensor,
                tac_feat: Tensor,
                gt_feat: Optional[Tensor] = None):
        """
        vis_feat: [B, 768]
        tac_feat: [B, 768]
        gt_feat: [B, 768]
        """

        # Query Transformation
        vis_attn_feat = vis_feat.unsqueeze(1)
        tac_attn_feat = tac_feat.unsqueeze(1)
        vis_attn_feat = vis_attn_feat + self.vis_attn(self.vis_norm(vis_attn_feat))
        
        for attn, norm in zip(self.tac_attns, self.tac_norms):
            tac_attn_feat = tac_attn_feat + attn(norm(tac_attn_feat))

        query_emb = self.crs_attn(self.crs_norm(tac_attn_feat), vis_attn_feat)
        query_emb = (query_emb + self.query_proj(self.proj_norm(query_emb))).squeeze(1)
        query_emb = query_emb / torch.norm(query_emb, dim=-1, keepdim=True)
        output = dict(query_emb=query_emb)
        
        # only for training query transformation module
        if gt_feat is not None:
            q2txt_sim = F.cosine_similarity(query_emb, gt_feat, dim=-1).mean()
            q2tac_sim = F.cosine_similarity(query_emb, tac_feat, dim=-1).mean()
            q2vis_sim = F.cosine_similarity(query_emb, vis_feat, dim=-1).mean()
            
            align_loss = (1 - q2txt_sim) + (1 - q2tac_sim) * 0.2
            stability_loss = self.diversity_loss(query_emb) * 0.1 + \
                             F.mse_loss(query_emb, gt_feat) * 10 + \
                             self.clip_loss(query_emb, tac_feat) * 0.1
            
            losses = {
                "align_loss": align_loss,
                "stability_loss": stability_loss,
                'q2txt_sim': q2txt_sim,
                'q2tac_sim': q2tac_sim,
                'q2vis_sim': q2vis_sim
            }
            return output, losses
        else:
            return output
