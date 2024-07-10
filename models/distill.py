import torch
import torch.nn as nn

from .ghiasi import Ghiasi, DistilledGhiasi
from .stylePredictor import StylePredictor, DistilledStylePredictor
import numpy as np
import sys
from os.path import join, dirname
from .net import Net as PerceptualLoss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Distiller(nn.Module):
    def __init__(self):
        super(Distiller,self).__init__()

        # create transformer and style predictor networks:
        self.ghiasi = Ghiasi()
        self.stylePredictor = StylePredictor()
        self.ghiasi.to(device)
        self.stylePredictor.to(device)

        self.distill_ghiasi = DistilledGhiasi()
        self.distill_stylepredictor = DistilledStylePredictor()

        for p in self.ghiasi.parameters():
            p.requires_grad = False
        for p in self.stylePredictor.parameters():
            p.requires_grad = False

        self.distill_ghiasi.to(device)
        self.distill_stylepredictor.to(device)


        # load checkpoints:
        checkpoint_ghiasi = torch.load(join(dirname(__file__),'checkpoints/checkpoint_transformer.pth'))
        checkpoint_stylepredictor = torch.load(join(dirname(__file__),'checkpoints/checkpoint_stylepredictor.pth'))
        # checkpoint_embeddings = torch.load(join(dirname(__file__),'checkpoints/checkpoint_embeddings.pth'))
        
        # load weights for ghiasi and stylePredictor, and mean / covariance for the embedding distribution:
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'],strict=False)
        self.stylePredictor.load_state_dict(checkpoint_stylepredictor['state_dict_stylepredictor'],strict=False)

        self.mse = nn.MSELoss()
        self.perceptual = PerceptualLoss().to(device)


    def forward(self, content, style):
        with torch.no_grad():
            t_style = self.stylePredictor(style)
            t_content = self.ghiasi(content, t_style)

        s_style = self.distill_stylepredictor(style)
        s_content = self.distill_ghiasi(content, t_style)

        return t_content, s_content, t_style, s_style


    def compute_loss(self, content, style):
        t_content, s_content, t_style, s_style = self.forward(content, style)

        loss_distill: torch.Tensor = self.mse(t_content, s_content) + self.mse(t_style, s_style)
        loss_perceptual_c, loss_perceptual_s, loss_perceptual_sl = self.perceptual(content, style, s_content)
        loss_perceptual: torch.Tensor = 0.2 * loss_perceptual_c + loss_perceptual_s + 0.3 * loss_perceptual_sl
        return loss_distill + 0.5*loss_perceptual
    
    def compute_loss(self, content, style, t_content, s_content, t_style, s_style):
        loss_distill: torch.Tensor = self.mse(t_content, s_content) + self.mse(t_style, s_style)
        loss_perceptual_c, loss_perceptual_s, loss_perceptual_sl = self.perceptual(content, style, s_content)
        loss_perceptual: torch.Tensor = 0.2 * loss_perceptual_c + loss_perceptual_s + 0.3 * loss_perceptual_sl
        return loss_distill + 0.5*loss_perceptual
    
    def save_ckpt(self, path):
        torch.save({
            'ghiasi': self.distill_ghiasi.state_dict(),
            'stylepredictor': self.distill_stylepredictor.state_dict(),
        }, path)


    def load_ckpt(self, path):
        checkpoint = torch.load(path)
        self.distill_ghiasi.load_state_dict(checkpoint['ghiasi'])
        self.distill_stylepredictor.load_state_dict(checkpoint['stylepredictor'])


    