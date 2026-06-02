import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

class ExpertSplitter(nn.Module):
    def __init__(self, dim, num_experts=3, tau=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.tau = tau
        self.router_x = nn.Linear(dim, num_experts)
        self.router_mean = nn.Linear(dim, num_experts)
        self.router_var = nn.Linear(dim, num_experts)
        self.proj_relu = nn.Linear(dim, dim, bias=True)
        self.proj_gelu = nn.Linear(dim, dim, bias=True)
        self.proj_elu = nn.Linear(dim, dim, bias=True)
    def forward(self, x):
        B, N, C = x.shape
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        logits_x = self.router_x(x)
        logits_mean = self.router_mean(mean)
        logits_var = self.router_var(var)
        logits_mean = logits_mean.expand(-1, N, -1)
        logits_var = logits_var.expand(-1, N, -1)
        combined_logits = logits_x + logits_mean + logits_var
        if self.training:
            weights = F.gumbel_softmax(combined_logits, tau=self.tau, hard=True, dim=-1)
        else:
            idx = combined_logits.argmax(dim=-1)
            weights = F.one_hot(idx, num_classes=self.num_experts).float()
        out_relu = self.proj_relu(F.relu(x))
        out_gelu = self.proj_gelu(F.gelu(x))
        out_elu = self.proj_elu(F.elu(x))
        w_relu = weights[..., 0].unsqueeze(-1)
        w_gelu = weights[..., 1].unsqueeze(-1)
        w_elu = weights[..., 2].unsqueeze(-1)
        out = w_relu * out_relu + w_gelu * out_gelu + w_elu * out_elu
        return out
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Fusion(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0.,
                 alpha=4.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim) ** -0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear2 = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop2 = nn.Dropout(proj_drop)

        self.scaleq = nn.Parameter(torch.zeros(1, 1, dim))
        self.scalek = nn.Parameter(torch.zeros(1, 1, dim))

        self.softmax = nn.Softmax(dim=-1)

        self.act_moe = ExpertSplitter(dim=dim)
        self.act_moe2 = ExpertSplitter(dim=dim)
        self.act_moe3 = ExpertSplitter(dim=dim)
        self.act_moe4 = ExpertSplitter(dim=dim)

    def _to_heads(self, x):
        B, L, C = x.shape
        return x.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
    def forward(self, x_q, x_k):
        B, Nq, C = x_q.shape
        B, Nk, C = x_k.shape

        q = self.q_linear(x_q)
        k = self.k_linear(x_k)
        vq = self.v_linear1(x_q)
        vk = self.v_linear2(x_k)

        scale_q = F.softplus(self.scaleq)
        scale_k = F.softplus(self.scalek)
        q = q / scale_q
        k = k / scale_k

        q_pos = self.act_moe(q)
        q_neg = self.act_moe2(-q)
        k_pos = self.act_moe3(k)
        k_neg = self.act_moe4(-k)

        q_pos = q_pos.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        q_neg = q_neg.reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k_pos = k_pos.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k_neg = k_neg.reshape(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        scale = self.scale
        # (B, H, Nq, Nk)
        attn_pp = (q_pos @ k_pos.transpose(-2, -1)) * scale
        attn_nn = (q_neg @ k_neg.transpose(-2, -1)) * scale
        attn_pn = (q_pos @ k_neg.transpose(-2, -1)) * scale
        attn_np = (q_neg @ k_pos.transpose(-2, -1)) * scale

        attn_sim_score = attn_pp + attn_nn
        attn_sim = self.softmax(attn_sim_score)

        attn_opp_score = attn_pn + attn_np
        attn_opp = self.softmax(attn_opp_score)

        v_q_heads = self._to_heads(vq)
        v_k_heads = self._to_heads(vk)

        x_sim_q = attn_sim.transpose(-2, -1) @ v_q_heads
        x_opp_q = attn_opp.transpose(-2, -1) @ v_q_heads
        x_sim_k = attn_sim @ v_k_heads
        x_opp_k = attn_opp @ v_k_heads

        y_q = x_sim_q + x_opp_q
        y_k = x_sim_k + x_opp_k

        y_q = y_q.transpose(1, 2).reshape(B, Nk, C)
        y_k = y_k.transpose(1, 2).reshape(B, Nq, C)

        y_q = self.proj(y_q)
        y_q = self.proj_drop(y_q)

        y_k = self.proj2(y_k)
        y_k = self.proj_drop2(y_k)

        return y_q, y_k

class FusionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=6., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,  use_power=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Fusion(dim, num_heads,qkv_bias=qkv_bias,attn_drop=0., proj_drop=0.)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)

    def forward(self, x_q, x_k):
        q, k = self.attn(x_q, x_k)
        #
        q = self.drop_path(q) + x_k
        k = self.drop_path(k) + x_q
        q = self.drop_path(self.mlp(self.norm1(q)))
        k = self.drop_path(self.mlp2(self.norm2(k)))

        q = q.mean(dim=1)
        k = k.mean(dim=1)
        q_k = torch.cat([q, k], dim=1)
        return q_k


