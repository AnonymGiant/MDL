

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention
from timm.models.layers import Mlp
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Expert(nn.Module):
    def __init__(self,dim=1024, mlp_ratio=.25, act_layer=nn.GELU, drop=0):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.net = nn.Sequential(

        #     nn.LayerNorm(dim),
        #     # nn.Linear(in_features=dim, out_features=dim, bias=False),
        #     nn.GELU(),
        #     # nn.Dropout(drop)
        #     # Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # )

        self.net = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=0)

        self.norm = nn.LayerNorm(dim)

    def forward(self,x):
        return self.norm(self.net(x) + x) 


class Allocator(nn.Module):
    def __init__(self, num_experts, top_k, expert_capacity, patchs_num, input_dim, expert_ratio, kernel_size=1):
        '''
        input_dim: [Batch, patchs_num, input_dim]，每一个visual token的维度
        '''
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity

        self.gate = nn.Linear(input_dim, num_experts)

        self.experts = nn.ModuleList(
            [Expert(input_dim, expert_ratio) for _ in range(num_experts)]
        )

        # self.attn = Attention(dim=input_dim, num_heads=8)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        batch_size, patchs_num, input_dim = x.shape
        device = x.device

        # [batch_szie, patchs_num, num_experts]
        logit = self.gate(x)

        # [batch_szie, patchs_num, num_experts]
        probs = torch.softmax(logit, dim=-1)


        # 对于一个patch来说，要放到哪些expert里
        # [batch_szie, patch_size**2, top_k]
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        # 重要性损失函数
        if self.training:
            importance = probs.sum(0)
            importance_loss = torch.var(importance) / (self.num_experts ** 2)

            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
            routing_probs = probs * mask
            expert_usage = mask.float().mean(0)
            routing_weights = routing_probs.mean(0)
            load_balance_loss = self.num_experts * (expert_usage * routing_weights).sum()

            aux_loss = importance_loss + load_balance_loss
        else:
            aux_loss = 0


        # [batch_szie * patch_size**2 * top_k]
        flat_idx = topk_idx.view(-1)
        flat_probs = topk_probs.view(-1)


        # 因为拉成一维了，所以需要对应一维上每个位置的patch，生成对应位置的样本索引
        # [batch_szie * patchs_num]
        sample_idx = torch.arange(batch_size * patchs_num, device=device)[:, None]
        # [batch_szie * patchs_num * top_k]
        sample_idx = sample_idx.expand(-1, self.top_k).flatten()

        # Batch indicex
        batch_idx = torch.arange(batch_size, device=device)[:, None]
        batch_idx = batch_idx.expand(-1, patchs_num * self.top_k).flatten()

        results = []
        for expert_idx in range(self.num_experts):
            expert_mask = flat_idx == expert_idx

            expert_samples = sample_idx[expert_mask]
            expert_weights = flat_probs[expert_mask]
            expert_batch = batch_idx[expert_mask]

            # if len(expert_samples) > self.expert_capacity:
            #     expert_samples = expert_samples[:self.expert_capacity]
            #     expert_weights = expert_weights[:self.expert_capacity]

            if len(expert_samples) == 0:
                expert_input = torch.randn(batch_size, input_dim).to(device) * 1e-5
                expert_output = self.experts[expert_idx](expert_input)
                results.append(expert_output.unsqueeze(0))
                continue

            expert_input = x.reshape(-1,input_dim)[expert_samples]
            expert_input = expert_input + torch.randn(expert_input.size(0), input_dim).to(device) * 1e-5

            expert_output = self.experts[expert_idx](expert_input)
            # weight_output = expert_output * expert_weights.unsqueeze(-1)
            weight_output = expert_output

            # 将expert_output分配到各个Batch上
            onehot = torch.nn.functional.one_hot(expert_batch, num_classes=batch_size).float() 
            sums = torch.mm(onehot.t(), weight_output) 

            results.append(sums.unsqueeze(0))
        
        x = torch.cat(results, dim=0).transpose(1,0)
        x = self.norm(x)

        return x, aux_loss

