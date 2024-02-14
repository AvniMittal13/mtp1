import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange
from scipy import signal
import numpy as np

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = torch.nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = torch.nn.Softmax(dim=-1)

        self.mask_heads = None
        self.attn_map = None

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim),
            torch.nn.Dropout(dropout)
        ) if project_out else torch.nn.Identity()

    def forward(self, x, localize=None, h=None, w=None, d=None, **kwargs):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if localize is not None:
            q = rearrange(q, 'b h n d -> b h n 1 d')
            k = localize(k, h, w, d)  # b h n (attn_height attn_width attn_depth) d
            v = localize(v, h, w, d)  # b h n (attn_height attn_width attn_depth) d

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # b h n 1 (attn_height attn_width attn_depth)

        attn = self.attend(dots)  # b h n 1 (attn_height attn_width attn_depth)

        if kwargs.get('mask', False):
            mask = kwargs['mask']
            assert len(mask) <= attn.shape[1], 'number of heads to mask must be <= number of heads'
            attn[:, mask] *= 0.0

        self.attn_maps = attn

        out = torch.matmul(attn, v)  # b h n 1 d
        out = rearrange(out, 'b h n 1 d -> b n (h d)') if localize else rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(torch.nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, head_dim=head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dim, dropout=dropout))
            ]))

    def encode(self, x, attn, ff, localize_attn_fn=None, h=None, w=None, d=None, **kwargs):
        x = attn(x, localize=localize_attn_fn, h=h, w=w, d=d, **kwargs) + x
        x = ff(x) + x
        return x

    def forward(self, x, localize_attn_fn=None, h=None, w=None, d=None, **kwargs):
        if self.training and len(self.layers) > 1:
            funcs = [lambda _x: self.encode(_x, attn, ff, localize_attn_fn, h, w, d, **kwargs) for attn, ff in
                     self.layers]
            x = torch.utils.checkpoint.checkpoint_sequential(funcs, segments=len(funcs), input=x)
        else:
            for attn, ff in self.layers:
                x = self.encode(x, attn, ff, localize_attn_fn, h, w, d, **kwargs)
        return x

class LocalizeAttention(torch.nn.Module):
    def __init__(self, attn_neighbourhood_size, device):
        super().__init__()
        self.attn_neighbourhood_size = attn_neighbourhood_size
        self.device = device
        self.attn_filters = neighbourhood_filters(self.attn_neighbourhood_size, self.device)

    def forward(self, x, height, width, depth):
        '''attn_filters: [filter_n, h, w, d]'''

        print("7: ", x.shape, height, width, depth)
        b, h, _, d = x.shape
        y = rearrange(x, 'b h (i j k) d -> (b h d) 1 i j k', i=height, j=width, k=depth)
        y = torch.nn.functional.conv3d(y, self.attn_filters[:, None], padding='same')
        _x = rearrange(y, '(b h d) filter_n i j k -> b h (i j k) filter_n d', b=b, h=h, d=d)
        return _x

def neighbourhood_filters(neighbourhood_size, device):
    height, width, depth = neighbourhood_size
    impulses = []
    for i in range(height):
        for j in range(width):
            for k in range(depth):
                impulse = signal.unit_impulse((height, width, depth), idx=(i, j, k), dtype=np.float32)
                impulses.append(impulse)
    filters = torch.tensor(np.stack(impulses), device=device)
    return filters



class BasicViTNCA3D(nn.Module):
    def __init__(self, channel_n, fire_rate, device, hidden_size=128, input_channels=1, init_method="standard", kernel_size=7, groups=False,
                 depth=1, heads=1, mlp_dim=64, dropout=0., embed_dim = 16):
        r"""Init function
            #Args:
                channel_n: number of channels per cell
                fire_rate: random activation of each cell
                device: device to run model on
                hidden_size: hidden size of model
                input_channels: number of input channels
                init_method: Weight initialisation function
                kernel_size: defines kernel input size
                groups: if channels in input should be interconnected
        """
        super(BasicViTNCA3D, self).__init__()

        self.device = device
        self.channel_n = channel_n
        self.input_channels = input_channels

        # One Input
        self.fc0 = nn.Linear(channel_n*2, hidden_size)
        self.fc1 = nn.Linear(hidden_size, channel_n, bias=False)
        padding = int((kernel_size-1) / 2)

        self.p0 = nn.Conv3d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=padding, padding_mode="reflect", groups=channel_n)
        self.bn = torch.nn.BatchNorm3d(hidden_size)
        
        with torch.no_grad():
            self.fc1.weight.zero_()

        if init_method == "xavier":
            torch.nn.init.xavier_uniform(self.fc0.weight)
            torch.nn.init.xavier_uniform(self.fc1.weight)

        self.fire_rate = fire_rate
        self.to(self.device)

        # rearranging from 3D grid to 1D sequence
        self.rearrange_cells = Rearrange('b h w d c -> b (h w d) c')
        self.localized_attn_neighbourhood = [3,3,3]
        self.localize_attn_fn = LocalizeAttention(self.localized_attn_neighbourhood, device)
        self.transformer = Transformer(embed_dim, depth, heads, embed_dim // heads, mlp_dim, dropout)

        self.mlp_head = torch.nn.Sequential(
			torch.nn.LayerNorm(embed_dim),
			torch.nn.Linear(embed_dim, self.channel_n)
		)

		# don't update cells before first backward pass or else cell grid will have immensely diverged and grads will
		# be large and unhelpful
        self.mlp_head[1].weight.data.zero_()
        self.mlp_head[1].bias.data.zero_()


    def positional_encoding_3d(self, depth, height, width, dim):
        pos_depth = torch.arange(depth, device= self.device).unsqueeze(1)
        pos_height = torch.arange(height, device= self.device).unsqueeze(1)
        pos_width = torch.arange(width, device= self.device).unsqueeze(1)
        
        div_term_depth = torch.exp(torch.arange(0, dim, 2, device= self.device) * (-math.log(10000.0) / dim))
        div_term_height = torch.exp(torch.arange(1, dim, 2, device= self.device) * (-math.log(10000.0) / dim))
        div_term_width = torch.exp(torch.arange(2, dim + 2, 2, device= self.device) * (-math.log(10000.0) / dim))
        
        pe_depth = torch.zeros(depth, 1, 1, dim, device= self.device)
        pe_height = torch.zeros(1, height, 1, dim, device= self.device)
        pe_width = torch.zeros(1, 1, width, dim, device= self.device)
        
        # print(pe_depth.shape, pos_depth * div_term_depth)
        if dim == 1:
            pe_depth[:, 0, 0, 0] = torch.sin(pos_depth.squeeze() * div_term_depth)
            pe_height[0, :, 0, 0] = torch.sin(pos_height.squeeze() * div_term_height)
            pe_width[0, 0, :, 0] = torch.sin(pos_width.squeeze() * div_term_width)
        else:

            pe_depth[:, 0, 0, 0::2] = torch.sin(pos_depth * div_term_depth)
            pe_depth[:, 0, 0, 1::2] = torch.cos(pos_depth * div_term_depth)
            
            pe_height[0, :, 0, 0::2] = torch.sin(pos_height * div_term_height)
            pe_height[0, :, 0, 1::2] = torch.cos(pos_height * div_term_height)
            
            pe_width[0, 0, :, 0::2] = torch.sin(pos_width * div_term_width)
            pe_width[0, 0, :, 1::2] = torch.cos(pos_width * div_term_width)
            
        pe = pe_depth + pe_height + pe_width
        return pe  # (depth, height, width, dim)


    def perceive(self, x):
        r"""Perceptive function, combines learnt conv outputs with the identity of the cell
            #Args:
                x: image
        """
        batch_size, height, width, depth, n_channels = x.size() 

        print("1: ", x.shape)
        # reshappe to (b (hwd) c)
        y = self.rearrange_cells(x)
        print("2: ", y.shape)
        # add position encoding to one channel
            # rearranginh 'b h w d c -> b (h w d) c'
        pe = self.positional_encoding_3d(height, width, depth, 2)
        pe = rearrange(pe, 'h w d c -> 1 (h w d) c')
        print("3: ", pe.shape)

        y[..., -2:] = pe
        print("4: ", y.shape)
        # y = self.dropout(y)
        print("5: ", y.shape)
        cells = rearrange(x, 'b h w d c -> b c h w d')
        y = self.transformer(y, localize_attn_fn=self.localize_attn_fn, h=cells.shape[-3], w=cells.shape[-2], d=cells.shape[-1])
        print("6: ", y.shape)


        # y1 = self.p0(x)
        # y = torch.cat((x,y1),1)

        return y

    def update(self, x_in, fire_rate):
        r"""Update function runs same nca rule on each cell of an image with a random activation
            #Args:
                x_in: image
                fire_rate: random activation of cells
        """
        # x = x_in.transpose(1,4)
        x = x_in
        dx = self.perceive(x_in)
        dx = self.mlp_head(dx)
        dx = rearrange(dx, 'b (h w d) c -> b h w d c', h=x_in.shape[1], w = x_in.shape[2], d = x_in.shape[3])

        print("8: ", dx.shape, x.shape)
        # dx = dx.transpose(1,4)
        # dx = self.fc0(dx)
        # dx = dx.transpose(1,4)
        # dx = self.bn(dx)
        # dx = dx.transpose(1,4)
        # dx = F.relu(dx)
        # dx = self.fc1(dx)

        if fire_rate is None:
            fire_rate=self.fire_rate
        stochastic = torch.rand([dx.size(0),dx.size(1),dx.size(2), dx.size(3),1])>fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        print("final: ", x.shape, dx.shape)

        # x = x+dx.transpose(1,4)
        x = x + dx

        # x = x.transpose(1,4)

        return x

    def forward(self, x, steps=10, fire_rate=0.5):
        r"""Forward function applies update function s times leaving input channels unchanged
            #Args:
                x: image
                steps: number of steps to run update
                fire_rate: random activation rate of each cell
        """
        for step in range(steps):
            x2 = self.update(x, fire_rate).clone() #[...,3:][...,3:]
            x = torch.concat((x[...,0:self.input_channels], x2[...,self.input_channels:]), 4)
        return x
