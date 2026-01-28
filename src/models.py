"""
Classification models for Quick, Draw! dataset experiments.
Includes MLP, ResNet-18, and Vision Transformer (ViT) architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from einops import rearrange, repeat
    from einops.layers.torch import Rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    print("Warning: einops not available. ViT models will not work. Install with: pip install einops")


# ============== MLP Classifier ==============

class MLPClassifier(nn.Module):
    """Simple Multi-Layer Perceptron for image classification.
    
    Args:
        input_channels: Number of input channels (default: 1 for grayscale Quick, Draw!)
        image_size: Input image size (default: 28)
        num_classes: Number of output classes (default: 345 for Quick, Draw!)
        hidden_dims: Tuple of hidden layer dimensions (default: (512, 256, 128))
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(
        self,
        input_channels=1,
        image_size=28,
        num_classes=345,
        hidden_dims=(512, 256, 128),
        dropout=0.2
    ):
        super().__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        in_dim = input_channels * image_size * image_size
        
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        feats = self.backbone(x)
        return self.head(feats)


# ============== ResNet-18 Classifier ==============

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18."""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 classifier adapted for small images (e.g., Quick, Draw! 28x28).
    
    Standard architecture from "Deep Residual Learning for Image Recognition":
    - Initial 3×3 conv (adapted for small images)
    - 4 stages with [2, 2, 2, 2] BasicBlocks
    - Channels: [64, 128, 256, 512]
    - Global average pooling + fc
    
    Args:
        num_classes: Number of output classes (default: 345 for Quick, Draw!)
        input_channels: Number of input channels (default: 1 for grayscale)
    """
    
    def __init__(self, num_classes=345, input_channels=1):
        super().__init__()
        
        # Initial convolution (adapted for small images: 3×3 instead of 7×7, no maxpool)
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # ResNet-18 stages: [2, 2, 2, 2] blocks
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ============== Vision Transformer (ViT) ==============

def pair(t):
    """Helper to convert int to tuple."""
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    """Layer normalization before the given function."""
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """MLP block with GELU activation."""
    
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
    """Multi-head self-attention."""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        if not EINOPS_AVAILABLE:
            raise RuntimeError("einops is required for Attention. Install with: pip install einops")
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer encoder with multiple layers."""
    
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification.
    
    Adapted for Quick, Draw! dataset (28x28 grayscale images).
    
    Args:
        image_size: Input image size (default: 28 for Quick, Draw!)
        patch_size: Size of image patches (default: 4)
        num_classes: Number of output classes (default: 345 for Quick, Draw!)
        dim: Embedding dimension (default: 256)
        depth: Number of transformer layers (default: 6)
        heads: Number of attention heads (default: 8)
        mlp_dim: Hidden dimension in MLP blocks (default: 512)
        pool: Pooling type: 'cls' or 'mean' (default: 'cls')
        channels: Number of input channels (default: 1 for grayscale)
        dim_head: Dimension per attention head (default: 64)
        dropout: Dropout rate (default: 0.)
        emb_dropout: Embedding dropout rate (default: 0.)
    """
    
    def __init__(
        self,
        *,
        image_size=28,
        patch_size=4,
        num_classes=345,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        pool='cls',
        channels=1,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        if not EINOPS_AVAILABLE:
            raise RuntimeError("einops is required for ViT. Install with: pip install einops")
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        return self.mlp_head(x)


# ============== Model Factory ==============

def create_model(model_type, num_classes=345, input_channels=1, image_size=28, **kwargs):
    """Factory function to create models by name.
    
    Args:
        model_type: One of 'mlp', 'resnet18', 'vit'
        num_classes: Number of output classes
        input_channels: Number of input channels
        image_size: Input image size
        **kwargs: Additional model-specific arguments
    
    Returns:
        PyTorch model
    """
    models = {
        'mlp': lambda: MLPClassifier(
            input_channels=input_channels,
            image_size=image_size,
            num_classes=num_classes,
            **kwargs
        ),
        'resnet18': lambda: ResNet18(
            num_classes=num_classes,
            input_channels=input_channels,
            **kwargs
        ),
        'vit': lambda: ViT(
            image_size=image_size,
            num_classes=num_classes,
            channels=input_channels,
            **kwargs
        ),
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type]()
