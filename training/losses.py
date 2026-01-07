# losses.py

import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#### Representation Orthogonality Constraint ####
class OrthogoDiffLoss(nn.Module):

    def __init__(self):
        super(OrthogoDiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
    
#### Shared Information Alignment Loss ####
class SharedInformationAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        """
        Initializes the SharedInformationAlignmentLoss.
        
        Args:
            temperature (float): Temperature parameter \( \tau \) for scaling similarities.
        """
        super(SharedInformationAlignmentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_img, z_txt, z_shared):
        """
        Computes the alignment loss.

        Args:
            z_img (torch.Tensor): Image embeddings, shape (batch_size, embedding_dim).
            z_txt (torch.Tensor): Text embeddings, shape (batch_size, embedding_dim).
            z_shared (torch.Tensor): Shared embeddings, shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: The alignment loss.
        """
        # Normalize the embeddings to unit vectors
        z_img = F.normalize(z_img, dim=1)
        z_txt = F.normalize(z_txt, dim=1)
        z_shared = F.normalize(z_shared, dim=1)

        # Compute similarities
        sim_s_i = torch.mm(z_shared, z_img.t()) / self.temperature  # (batch_size, batch_size)
        sim_s_t = torch.mm(z_shared, z_txt.t()) / self.temperature  # (batch_size, batch_size)

        # Create ground-truth labels for similarity (diagonal should match across batches)
        labels = torch.arange(z_shared.size(0), device=z_shared.device)

        # Compute cross-entropy losses
        loss_s_i = F.cross_entropy(sim_s_i, labels)
        loss_s_t = F.cross_entropy(sim_s_t, labels)

        # Combine losses
        loss = loss_s_i + loss_s_t
        return loss
    
class ContrastiveAlignmentLoss(nn.Module):
    """
    Calculates the InfoNCE loss for contrastive alignment between two modalities.
    It encourages positive pairs (matching visual and text) to have higher similarity
    than negative pairs.
    """
    def __init__(self, temperature=0.1):
        """
        Args:
            temperature (float): Scales the logits to control the sharpness of the
                                 probability distribution.
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, visual_embeds, text_embeds):
        """
        Args:
            visual_embeds (torch.Tensor): A batch of global visual embeddings.
                                          Shape: (batch_size, embed_dim).
            text_embeds (torch.Tensor): A batch of global text embeddings.
                                        Shape: (batch_size, embed_dim).
        
        Returns:
            torch.Tensor: The calculated contrastive loss.
        """
        # 1. Normalize the embeddings to unit vectors
        visual_embeds = F.normalize(visual_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        # 2. Calculate the cosine similarity matrix
        # The dot product of unit vectors is the cosine similarity.
        # Shape: (batch_size, batch_size)
        similarity_matrix = torch.matmul(visual_embeds, text_embeds.T) / self.temperature

        # 3. Create ground-truth labels. The positive pair for image `i` is text `i`.
        batch_size = visual_embeds.shape[0]
        labels = torch.arange(batch_size, device=visual_embeds.device)

        # 4. Calculate loss in both directions (image-to-text and text-to-image)
        loss_i2t = self.criterion(similarity_matrix, labels)
        loss_t2i = self.criterion(similarity_matrix.T, labels) # Transpose for text-to-image

        # 5. Average the two losses
        return (loss_i2t + loss_t2i) / 2.0

class RKDLoss(nn.Module):
    """Relational Knowledge Distillation, CVPR2019"""
    def __init__(self, w_d=25.0, w_a=50.0):
        super(RKDLoss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def pdist(self, e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def forward(self, f_s, f_t):
        with torch.no_grad():
            t_d = self.pdist(f_t, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(f_s, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d, reduction='mean')

        with torch.no_grad():
            t_d_pairs = f_t.unsqueeze(1) - f_t.unsqueeze(0)
            norm_td = F.normalize(t_d_pairs, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2))

        s_d_pairs = f_s.unsqueeze(1) - f_s.unsqueeze(0)
        norm_sd = F.normalize(s_d_pairs, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2))

        loss_a = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        loss = self.w_d * loss_d + self.w_a * loss_a
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, device):
        super(TripletLoss, self).__init__()
        self.device = device
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.matmul(image_features, text_features.transpose(0, 1))
        logits1 = logit_scale * torch.matmul(text_features, image_features.transpose(0, 1))
        labels = torch.tensor(np.arange(12)).to(self.device)
        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits1, labels)
        loss = loss1 + loss2
        return loss

class HSIC(nn.Module):
    """
    Computes the unbiased Hilbert-Schmidt Independence Criterion (HSIC) loss.
    
    This loss function measures the statistical independence between two sets of
    random variables. A value of 0 indicates statistical independence. This
    implementation uses the RBF (Gaussian) kernel.
    
    Reference:
    Gretton, A., et al. (2007). A kernel statistical test of independence. NIPS.
    """
    def __init__(self, sigma=1.0):
        """
        Args:
            sigma (float): The bandwidth parameter for the RBF kernel. A common
                           heuristic is to set this to the median pairwise
                           distance between points.
        """
        super(HSIC, self).__init__()
        self.sigma = sigma

    def _rbf_kernel(self, x, y):
        """
        Computes the RBF (Gaussian) kernel matrix between two tensors.
        
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, feature_dim).
            y (torch.Tensor): A tensor of shape (batch_size, feature_dim).
            
        Returns:
            torch.Tensor: The kernel matrix of shape (batch_size, batch_size).
        """
        # Ensure inputs are 2D
        if x.dim() == 1: x = x.unsqueeze(1)
        if y.dim() == 1: y = y.unsqueeze(1)
            
        beta = 1.0 / (2.0 * self.sigma**2)
        # Compute pairwise squared Euclidean distances
        dist_sq = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-beta * dist_sq)

    def forward(self, x, y):
        """
        Calculates the HSIC statistic.
        
        Args:
            x (torch.Tensor): A tensor of shape (batch_size, feature_dim).
            y (torch.Tensor): A tensor of shape (batch_size, feature_dim).
            
        Returns:
            torch.Tensor: The HSIC loss value (a scalar tensor).
        """
        if x.shape[0] != y.shape[0]:
            raise ValueError("Input tensors must have the same batch size.")
        
        m = x.shape[0]
        if m < 2:
            # HSIC is not well-defined for batch size < 2
            return torch.tensor(0.0, device=x.device)

        # Step 1: Compute the RBF kernel matrices for x and y
        Kx = self._rbf_kernel(x, x)
        Ky = self._rbf_kernel(y, y)

        # Step 2: Create the centering matrix H
        # H = I - (1/m) * 1 * 1^T
        H = torch.eye(m, device=x.device) - 1.0 / m

        # Step 3: Compute the HSIC statistic using the unbiased estimator
        # The formula is: hsic = (1 / (m-1)^2) * tr(Kx * H * Ky * H)
        # Note: The original implementation you provided is correct.
        hsic_value = torch.trace(Kx @ H @ Ky @ H) / ((m - 1) ** 2)
        
        return hsic_value