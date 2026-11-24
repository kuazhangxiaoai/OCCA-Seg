import torch
import torch.nn as nn
from torchvision.models import (
    convnext_large,
    convnext_base,
    convnext_small,
    convnext_tiny,
    swin_b,
    swin_v2_b,
    swin_v2_s,
    swin_v2_t,
    mobilenet_v3_large,
    efficientnet_v2_m,
)
import torchvision.models as models
import torch.nn.functional as F

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def window_partition(x, window_size, channel_first=True):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    #x = x.permute(0, 2, 3, 1).contiguous() if channel_first else x
    if channel_first:
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
    else:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, channel_first=True):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C) or (num_windows*B, C, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C) or (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    if channel_first:
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))


class BaseNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        if self.model == "swin_base":
            swin_v2 = swin_b(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_base":
            swin_v2 = swin_v2_b(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_small":
            swin_v2 = swin_v2_s(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "swinv2_tiny":
            swin_v2 = swin_v2_t(weights="IMAGENET1K_V1")
            self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_large":
            convnext = convnext_large(pretrained=True)
            self.backbone = torch.nn.Sequential(*list(convnext.children())[:-1])
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_base":
            convnext = convnext_base(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_small":
            convnext = convnext_small(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "convnext_tiny":
            convnext = convnext_tiny(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(convnext.children())[:-1]))
            self.target_layer_names = ["0.1", "0.3", "0.5", "0.7"]
            self.multi_scale_features = []

        if self.model == "resnet":
            resnet101 = models.resnet101(pretrained=True)
            self.backbone = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
            self.target_layer_names = ["4", "5", "6", "7"]
            self.multi_scale_features = []

        if self.model == "mobilenet":
            mobilenet = mobilenet_v3_large(pretrained=True).features
            self.backbone = mobilenet
            self.target_layer_names = ["3", "6", "12", "16"]
            self.multi_scale_features = []

        if self.model == "efficientnet":
            efficientnet = efficientnet_v2_m(pretrained=True).features
            self.backbone = efficientnet
            self.target_layer_names = ["2", "3", "5", "8"]
            self.multi_scale_features = []

        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))

        num_encoder_blocks = 4
        if self.model in ["swin_base", "swinv2_base", "convnext_base"]:
            hidden_sizes = [128, 256, 512, 1024]
        if self.model in ["convnext_large"]:
            hidden_sizes = [192, 384, 768, 1536]
        if self.model in ["resnet"]:
            hidden_sizes = [256, 512, 1024, 2048]
        if self.model in [
            "swinv2_small",
            "swinv2_tiny",
            "convnext_small",
            "convnext_tiny",
        ]:
            hidden_sizes = [
                96,
                192,
                384,
                768,
            ]
        if self.model in ["mobilenet"]:
            hidden_sizes = [24, 40, 112, 960]
        if self.model in ["efficientnet"]:
            hidden_sizes = [48, 80, 176, 1280]
        self.hidden_sizes = hidden_sizes

class MSF(nn.Module):
    """
        Multiscale fusion 
    """
    def __init__(self):
        super().__init__()

    def forward(self, feats):
        """
            feats (list[Tensor]): features from backbone.
        """
        _, _, h, w = feats[0].shape
        flist = [F.interpolate(f, size=(h, w), mode="bilinear", align_corners=True) for f in feats]
        return torch.cat(flist, dim=1)

class OCAM(nn.Module):
    def __init__(self, cls_num=0, scale=1):
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs):
        k = feats.size(1) # []
        b, c, h, w = probs.shape # [batchsize, cls_num, height, width]

        feats = feats.view(b, k, -1).permute(0, 2, 1).contiguous()
        probs = probs.view(b, c, -1)
        probs = F.softmax(self.scale * probs, dim=2)

        ctx = torch.matmul(probs, feats).permute(0, 2, 1).contiguous().unsqueeze(3)
        return ctx

class CCAM(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, cls_num=0, scale=1, window_size=4, bias=True, drop=0.1, windows_part=True, use_qkv=True, shortcut=False):
        super().__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)
        self.hidden_dim = hidden_dim
        self.windows_part = windows_part
        self.use_qkv = use_qkv
        self.window_size = window_size
        self.shortcut = shortcut
        self.pre_conv = Conv(dim, out_dim, act=False) if shortcut else None

        if windows_part and use_qkv:
            self.query_layer = nn.Sequential(
                nn.Linear(dim * window_size * window_size, hidden_dim * window_size * window_size, bias),
                nn.Dropout(drop),
                nn.LayerNorm(hidden_dim * window_size * window_size))
            self.key_layer = nn.Sequential(
                nn.Linear(dim * window_size * window_size, cls_num * window_size * window_size, bias),
                nn.Dropout(drop),
                nn.LayerNorm(cls_num * window_size * window_size))
            self.value_layer = nn.Sequential(
                nn.Linear(cls_num * window_size * window_size, cls_num * window_size * window_size, bias),
                nn.Dropout(drop),
                nn.LayerNorm(cls_num * window_size * window_size))
            #self.fc_out = nn.Sequential(
            #    nn.Linear(hidden_dim * window_size * window_size, dim * window_size * window_size, bias),
            #    nn.GELU(),
            #    nn.Dropout(drop),
            #    nn.LayerNorm(dim * window_size * window_size))
            self.final_conv = Conv(hidden_dim, out_dim)
        elif use_qkv and (not windows_part):
            self.query_layer = nn.Sequential(nn.Linear(dim, hidden_dim, bias),
                                             nn.Dropout(drop),
                                             nn.LayerNorm(hidden_dim))
            self.key_layer = nn.Sequential(nn.Linear(dim, cls_num, bias),
                                           nn.Dropout(drop),
                                           nn.LayerNorm(cls_num))
            self.value_layer = nn.Sequential(nn.Linear(cls_num, cls_num, bias),
                                             nn.Dropout(drop),
                                             nn.LayerNorm(cls_num))
            #self.fc_out = nn.Sequential(nn.Linear(hidden_dim, dim, bias),
            #                            nn.GELU(),
            #                            nn.Dropout(drop),
            #                            nn.LayerNorm(dim))
            self.final_conv = Conv(hidden_dim, out_dim)
        else:
            self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim, bias), nn.Dropout(drop), nn.LayerNorm(hidden_dim))
            self.fc2 = nn.Sequential(nn.Linear(cls_num, cls_num, bias), nn.Dropout(drop), nn.LayerNorm(cls_num))
            self.final_conv = Conv(cls_num * hidden_dim, out_dim)

    def forward(self, feats, probs):
        if self.shortcut:
            x = self.pre_conv(feats)
        if self.use_qkv:
            ib, ic, ih, iw = feats.shape
            if self.windows_part:
                win_feats = window_partition(feats, self.window_size, True)
                win_probs = window_partition(probs, self.window_size, True)
                b, c, h, w = win_probs.shape
                k = win_feats.size(1)
                feats = win_feats.permute(0, 2, 3, 1).contiguous().view(b, k * h * w)
                probs = win_probs.permute(0, 2, 3, 1).contiguous().view(b, c * h * w)
            else:
                b, c, h, w = probs.shape
                k = feats.size(1)
                probs = probs.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
                feats = feats.permute(0, 2, 3, 1).contiguous().view(b * h * w, k)

            query = self.query_layer(feats).unsqueeze(2)
            key = self.key_layer(feats).unsqueeze(1)
            value = self.value_layer(probs).unsqueeze(-1)

            qk_dot = torch.matmul(query, key)
            qk_dot = F.softmax(self.scale * qk_dot, dim=-1)
            if self.windows_part:
                ctx = torch.matmul(qk_dot, value).view(b , self.hidden_dim * h * w)
            else:
                ctx = torch.matmul(qk_dot, value).view(b, h * w, self.hidden_dim)
            #ctx = self.fc_out(ctx).view(ib, ic, ih, iw)
            ctx = ctx.permute(0, 2, 1).contiguous().view(ib, self.hidden_dim, ih, iw)
        else:
            b, cls_num, h, w = probs.shape
            c = feats.size(1)
            feats = feats.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
            probs = probs.permute(0, 2, 3, 1).contiguous().view(b * h * w, cls_num)
            feats = self.fc1(feats)
            probs = self.fc2(probs)
            probs = F.softmax(self.scale * probs, dim=-1)
            ctx = torch.matmul(feats.unsqueeze(2), probs.unsqueeze(1)).view(b, h * w, self.cls_num * self.hidden_dim)
            #ctx = self.fc_out(ctx.view(b * h * w, self.cls_num * self.hidden_dim)).view(b, h *w, self.cls_num *c)
            ctx = ctx.permute(0, 2, 1).contiguous().view(b, self.cls_num * self.hidden_dim, h, w)

        return  x + self.final_conv(ctx) if self.shortcut else self.final_conv(ctx)


class AM(nn.Module):
    """
        Aggregation module (AM) is used for fusion of object-level and channel-wise contextual information 
    """
    def __init__(self, in_channels, k_channels, scale=1):
        """
            initialize the Aggregation module.
            in_channels (int): the number of input feature channels.
            k_channels (int): the number of hidden feature channels.
            scale (float): the number of scale
        """
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.k_channels = k_channels
        self.maxpool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.fp = nn.Sequential(
            Conv(self.in_channels, self.k_channels, 1, 1, 0, act=False),
            Conv(self.k_channels, self.k_channels, 1, 1, 0, act=False)
        )
        self.fo = nn.Sequential(
            Conv(self.in_channels, self.k_channels, 1, 1, 0, act=False),
            Conv(self.k_channels, self.k_channels, 1, 1, 0, act=False)
        )

        self.down = Conv(self.in_channels, self.k_channels, 1, 1, 0, act=False)
        self.up = Conv(self.k_channels, self.in_channels, 1, 1, 0, act=False)

    def forward(self, x, proxy, channel):
        b, _, h, w = x.shape
        if self.scale > 1:
            x = self.maxpool(x)
        query = self.fp(x).view(b, self.k_channels, -1).permute(0, 2, 1)
        key = self.fo(proxy).view(b, self.k_channels, -1)
        value = self.down(proxy).view(b, self.k_channels, -1).permute(0, 2, 1)

        _sim = F.softmax((self.k_channels ** -0.5) * torch.matmul(query, key), dim=-1)

        #bg context
        ctx = torch.matmul(_sim, value).permute(0, 2, 1).contiguous().view(b, self.k_channels, *x.size()[2:]) * channel
        ctx = self.up(ctx)
        if self.scale > 1:
            ctx = F.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=True)

        return ctx

class ContextGather(nn.Module):
    """
        Warpper of Aggregation module
    """
    def __init__(self, in_channels, k_channels, out_channels, scale=1, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.k_channels = k_channels
        self.out_channels = out_channels
        self.scale = scale

        self.am = AM(in_channels, k_channels, scale)

        self.conv= nn.Sequential(
            Conv(self.in_channels * 2, self.out_channels, 1, p=0),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy, channel):
        ctx = self.am(feats, proxy, channel)
        out = self.conv(torch.cat([ctx, feats], 1))
        return out

class OCCASegHead(nn.Module):
    """
        OCCA-Seg head for per-pixel classification.

        This head produces dense per-pixel class predictions. Unlike instance segmentation, no bounding boxes or instance
        masks are produced.

        Attributes:
            nc (int): Number of semantic classes.
            ns (int): Number of scale ratio to upsample.
            npr(int): Number of hidden channel dimension.
            ch (int): Number of input feature channels.

    """
    def __init__(self, nc=80, ns=4, npr=256, ch=()):
        """
            Initialize the semantic segmentation head.
            Args:
                nc (int): Number of semantic classes.
                ch (tuple): Tuple of channel sizes from neck feature maps (P3, P4).
        """
        super().__init__()
        self.nc = nc
        self.ns = ns
        self.npr = npr
        self.chs = torch.tensor(ch).sum().item()
        self.conv = nn.Conv2d(self.chs, self.npr * 2, kernel_size=3, stride=1, padding=1)
        self.msf = MSF()
        self.cls_head = nn.Conv2d(self.npr * 2, self.nc, kernel_size=1, stride=1, padding=0)

        self.norm_head = nn.Sequential(
            Conv(self.chs, self.chs, 3, 1, 1),
            nn.Conv2d(self.chs, self.nc, 1, 1, 0)
        )

        self.element_attn = OCAM(self.nc)
        self.channel_attn = CCAM(self.npr * 2, self.npr, self.npr, self.nc, use_qkv=False, windows_part=False)
        self.context_gather = ContextGather(2 * self.npr, self.npr, self.npr * 2, scale=1, dropout=0.05)

    def forward(self, x):
        fs = self.msf(x)
        h, w = fs.shape[2:]
        out0 = self.norm_head(fs)
        fs = self.conv(fs)
        ctx = self.element_attn(fs, out0)
        chl = self.channel_attn(fs, out0)
        feats = self.context_gather(fs, ctx, chl)
        out = self.cls_head(feats)
        out0 = F.interpolate(out0, size=(h*self.ns, w*self.ns), mode="bilinear", align_corners=True)
        out = F.interpolate(out, size=(h * self.ns, w * self.ns), mode="bilinear", align_corners=True)
        if self.training:
            return out0, out
        else:
            return out

class OCCASeg(BaseNet):
    def __init__(self,
                 model: str,
                 nc: int=80,
                 ns: int=8,
                 npr:int=256,
                 ch:tuple=(),
                 training:bool=True):
        super().__init__(model)
        self.training = training
        self.decoder = OCCASegHead(nc, ns, npr, ch)

    def save_features_hook(self, name):
        def hook(module, input, output):
            if self.model in [
                "swin_base",
                "swinv2_base",
                "swinv2_small",
                "swinv2_tiny",
            ]:
                self.multi_scale_features.append(
                    output.permute(0, 3, 1, 2).contiguous()
                )
            if self.model in [
                "convnext_large",
                "convnext_base",
                "convnext_small",
                "convnext_tiny",
                "mobilenet",
                "efficientnet",
                "resnet",
            ]:
                self.multi_scale_features.append(
                    output
                )

        return hook


    def forward(self, x):
        self.multi_scale_features.clear()
        _, _, h, w = x.shape
        self.backbone(x).squeeze() #[128, 256, 512, 1024]
        output = self.decoder(self.multi_scale_features)
        return output

if __name__ == "__main__":
    device = "cuda"
    num_classes = 7

    net = OCCASeg(
        #model="convnext_base",
        #ch=(128, 256, 512, 1024),
        model="convnext_large",
        ch=(192, 384, 768, 1536),
        nc=num_classes,
        ns=4,
        training=True
    ).to(device)

    fake_img = torch.randn(1,3,512,512).to(device)
    out = net(fake_img)
    for i in out:
        print(i.shape)