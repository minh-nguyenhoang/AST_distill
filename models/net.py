import torch.nn as nn
import torch
from typing import Tuple
import torch.nn.functional as F


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

vgg.load_state_dict(torch.hub.load_state_dict_from_url(
    'https://github.com/weepiess/StyleFlow-Content-Fixed-I2I/raw/master/vgg_model/vgg_normalised.pth', map_location='cpu'))


class Net(nn.Module):
    def __init__(self, encoder = vgg, max_sample = 64*64):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.mse_loss = nn.MSELoss()
        self.max_sample = max_sample

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_mean_std(self, feat: torch.Tensor, eps=1e-5) -> "Tuple[torch.Tensor, torch.Tensor]":
        size = feat.shape
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
    
    def mean_variance_norm(self, feat: torch.Tensor):
        size = feat.size()
        mean, std = self.calc_mean_std(feat)
        normalized_feat = (feat - mean.expand(size)) / std.expand(size)
        return normalized_feat

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)

        loss_gloabal = self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)



        return loss_gloabal

    def forward(self, content_images, style_images, stylized_images):
        style_feats = self.encode_with_intermediate(style_images)
        content_feats = self.encode_with_intermediate(content_images)
        stylized_feats = self.encode_with_intermediate(stylized_images)

        loss_c = self.calc_content_loss(stylized_feats[-1], content_feats[-1])
        loss_s = self.calc_style_loss(stylized_feats[0], style_feats[0])
        loss_ls = self.deterministic_adaattn(content_feats[0], style_feats[0], stylized_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(stylized_feats[i], style_feats[i])
            loss_ls += self.deterministic_adaattn(content_feats[i], style_feats[i], stylized_feats[i])
        return loss_c, loss_s, loss_ls
    
    def deterministic_adaattn(self, content, style, stylized):
        self.head = 1
        q: torch.Tensor = self.mean_variance_norm(content)
        k: torch.Tensor = self.mean_variance_norm(style)
        v: torch.Tensor = style
        b, _, h_g, w_g = k.size()
        b, _, h, w = q.size()
        if w_g * h_g > self.max_sample:
            k: torch.Tensor = F.interpolate(k.view(b, -1, w_g * h_g), size= self.max_sample, mode= 'nearest')
            style_flat: torch.Tensor = F.interpolate(v.view(b, -1, w_g * h_g), size= self.max_sample, mode= 'nearest')

            k = k.view(b, self.head, -1, self.max_sample).contiguous()
            style_flat = style_flat.view(b, self.head, -1, self.max_sample).transpose(-1, -2).contiguous()
        else:
            k = k.view(b, self.head, -1, w_g * h_g).contiguous()
            style_flat = v.view(b, self.head, -1, w_g * h_g).transpose(-1, -2).contiguous()  # v: b, head, n_s, c

        if w * h > self.max_sample:
            q = F.interpolate(q.view(b, -1, w * h), size= self.max_sample, mode= 'nearest')
            q = q.view(b, self.head, -1, self.max_sample).permute(0, 1, 3, 2)
        else:
            q = q.view(b, self.head, -1, w * h).permute(0, 1, 3, 2)
        
        S = torch.matmul(q, k)
        
        S = torch.softmax(S, dim= -1)  # S: b, head, n_c', n_s'
        
        mean = torch.matmul(S, style_flat) # mean: b, head, n_c', c
        
        std = torch.sqrt(torch.relu(torch.matmul(S, style_flat ** 2) - mean ** 2)) # std: b, head, n_c', c
        if w * h > self.max_sample:
            mean = mean.permute(0, 1, 3, 2).contiguous().view(b, -1, self.max_sample)
            mean = F.interpolate(mean, h*w, mode= "nearest").view(b, -1, h, w)
            std = std.permute(0, 1, 3, 2).contiguous().view(b, -1, self.max_sample)  
            std = F.interpolate(std, h*w, mode= "nearest").view(b, -1, h, w)
        else:
            mean = mean.permute(0, 1, 3, 2).contiguous().view(b, -1, h, w) # mean: b, c, h, w
            std = std.permute(0, 1, 3, 2).contiguous().view(b, -1, h, w)   # std : b, c, h, w            
        return self.mse_loss(stylized, std * self.mean_variance_norm(content) + mean) 