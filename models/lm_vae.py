import torch
import torch.nn as nn
import torch.nn.functional as F

# --- LatentEncoder class remains the same ---
class LatentEncoder(nn.Module):
    """
    A simple MLP that takes a feature vector and outputs the parameters (mu and log_var)
    of a Gaussian distribution.
    """
    def __init__(self, config, input_dim):
        super(LatentEncoder, self).__init__()
        self.fc_mu = nn.Linear(input_dim, config.latent_dim)
        self.fc_log_var = nn.Linear(input_dim, config.latent_dim)

    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

# --- GatingNetwork class as defined above ---
class GatingNetwork(nn.Module):
    """
    A simple MLP that acts as a gating mechanism for a Mixture of Experts.
    It takes combined features from all experts and outputs normalized weights.
    """
    def __init__(self, config, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.layer = nn.Sequential(
            nn.Linear(input_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 2, num_experts)
        )

    def forward(self, x):
        logits = self.layer(x)
        weights = F.softmax(logits, dim=-1)
        return weights

# --- Updated VAE class ---
class LongitudinalMultiModalVAE(nn.Module):
    def __init__(self, config):
        super(LongitudinalMultiModalVAE, self).__init__()
        self.config = config

        encoder_input_dim = config.embed_dim * 2

        # --- VAE Latent Encoders (Experts) ---
        self.vision_shared_encoder = LatentEncoder(config, encoder_input_dim)
        self.language_shared_encoder = LatentEncoder(config, encoder_input_dim)
        self.vision_specific_encoder = LatentEncoder(config, encoder_input_dim)
        self.language_specific_encoder = LatentEncoder(config, encoder_input_dim)

        # --- Gating Network for Shared Space ---
        # The input is the concatenation of both vision and language features
        gating_input_dim = encoder_input_dim * 2 # (v_p, v_c) + (l_p, l_c)
        self.gating_network = GatingNetwork(config, gating_input_dim, num_experts=2)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, v_p, v_c, l_p, l_c):
        # --- 1. Pool Embeddings ---
        v_p_pool = torch.mean(v_p, dim=1)
        v_c_pool = torch.mean(v_c, dim=1)
        l_p_pool = torch.mean(l_p, dim=1)
        l_c_pool = torch.mean(l_c, dim=1)

        # --- 2. Prepare Features for Encoders ---
        vision_features = torch.cat((v_p_pool, v_c_pool), dim=1)
        language_features = torch.cat((l_p_pool, l_c_pool), dim=1)

        # --- 3. Encode into Latent Distributions (Get Expert Opinions) ---
        mu_sh_v, log_var_sh_v = self.vision_shared_encoder(vision_features)
        mu_sh_l, log_var_sh_l = self.language_shared_encoder(language_features)
        mu_v_sp, log_var_v_sp = self.vision_specific_encoder(vision_features)
        mu_l_sp, log_var_l_sp = self.language_specific_encoder(language_features)

        # --- 4. Dynamically Combine Shared Experts using Gating Network ---
        # The gating network sees all available information to make its decision
        gating_input = torch.cat((vision_features, language_features), dim=1)
        expert_weights = self.gating_network(gating_input) # Shape: [B, 2]

        # Unsqueeze weights for broadcasting: [B, 2] -> [B, 1, 2]
        expert_weights = expert_weights.unsqueeze(1)

        # Stack expert parameters for weighted averaging: [B, D_latent] -> [B, D_latent, 2]
        all_mu_sh = torch.stack([mu_sh_v, mu_sh_l], dim=2)
        all_log_var_sh = torch.stack([log_var_sh_v, log_var_sh_l], dim=2)

        # Perform the weighted average (matrix multiplication)
        # [B, 1, 2] @ [B, D_latent, 2]^T -> [B, 1, D_latent] -> [B, D_latent]
        mu_sh = torch.bmm(expert_weights, all_mu_sh.transpose(1, 2)).squeeze(1)
        log_var_sh = torch.bmm(expert_weights, all_log_var_sh.transpose(1, 2)).squeeze(1)

        # --- 5. Sample from Latent Distributions ---
        z_sh = self.reparameterize(mu_sh, log_var_sh)
        z_v_sp = self.reparameterize(mu_v_sp, log_var_v_sp)
        z_l_sp = self.reparameterize(mu_l_sp, log_var_l_sp)

        z_sh_seq = z_sh.unsqueeze(1)
        z_v_sp_seq = z_v_sp.unsqueeze(1)
        z_l_sp_seq = z_l_sp.unsqueeze(1)

        # --- 6. Fuse Latent Variables and Decode ---
        z_fused = torch.cat((z_sh_seq, z_v_sp_seq, z_l_sp_seq), dim=1)

        return (z_fused, z_sh, z_v_sp, z_l_sp,
                mu_sh_v, log_var_sh_v, 
                mu_sh_l, log_var_sh_l,
                mu_v_sp, log_var_v_sp,
                mu_l_sp, log_var_l_sp)