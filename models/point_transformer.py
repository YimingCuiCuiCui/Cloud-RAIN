import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ii in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class point_ViT_cls(nn.Module):
    def __init__(self, point_size, num_classes=40, dim=256, depth=6, heads=8, mlp_dim=512,
                 pool='mean', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        num_patches = point_size
        patch_dim = channels
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, _, n = x.shape
        x = self.to_patch_embedding(x.permute(0,2,1))

        x = self.dropout(x)

        x = self.transformer(x)

        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = torch.max(x, dim=1)[0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class point_ViT_semseg(nn.Module):
    def __init__(self, point_size, num_classes=13, dim=256, depth=6, heads=8, mlp_dim=512, pool='mean',
                 channels=9, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        num_patches = point_size
        patch_dim = channels
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(nn.Linear(patch_dim, dim))

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        b, _, n = x.shape
        x = self.to_patch_embedding(x.permute(0,2,1))

        # cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        x = x + self.pos_embedding.repeat(b, 1, 1)
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_latent(x)
        return self.mlp_head(x).permute(0,2,1)

if __name__ == '__main__':
    model = point_ViT_cls(1024)
    pc = torch.randn(32, 3, 1024)
    pred = model(pc)
    print(pc.shape, pred.shape)
