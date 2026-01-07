from networkx import maximum_flow_value
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import CNNEncoder, TransformerEncoder, VisionEncoder, LanguageEncoder, Adaptor, TemoporalFactorizer, LongitudinalFusionNetwork, MultiModalFusionNetwork, LLaMAXDecoder, LanguageDecoder

class MoEFusionNetwork(nn.Module):
    """
    A Mixture of Experts (MoE)-style fusion network.

    This module takes multiple streams of sequential features and learns to
    dynamically weight them before fusing them into a single output stream.
    """
    def __init__(self, config, num_experts=3):
        super(MoEFusionNetwork, self).__init__()
        self.config = config
        self.num_experts = num_experts

        # The gating network decides the weights for each expert.
        # It looks at the pooled (mean) representation of all input streams.
        gating_input_dim = config.embed_dim * num_experts
        self.gating_network = nn.Sequential(
            nn.Linear(gating_input_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim // 2, num_experts)
        )

    def forward(self, *expert_features):
        """
        Args:
            *expert_features: A variable number of tensors, each of shape [B, Seq_i, D_embed].
                              In this case, it will be (mmfo_sh, mmfo_sp, r_p).
        """
        if len(expert_features) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} expert features, but got {len(expert_features)}")

        # --- 1. Gating Mechanism ---
        # Create a global representation for the gating network by pooling each expert.
        pooled_features = [torch.mean(feat, dim=1) for feat in expert_features]
        gating_input = torch.cat(pooled_features, dim=1)

        # Get the weights from the gating network (shape: [B, num_experts])
        expert_weights = F.softmax(self.gating_network(gating_input), dim=-1)

        # --- 2. Apply Weights and Fuse ---
        weighted_features = []
        for i, feat in enumerate(expert_features):
            # Get the weight for the current expert (shape: [B])
            weight = expert_weights[:, i]
            # Reshape weight for broadcasting: [B] -> [B, 1, 1]
            weight = weight.unsqueeze(1).unsqueeze(2)
            # Scale the entire feature tensor by its learned weight
            weighted_features.append(feat * weight)

        # Concatenate the weighted features along the sequence dimension,
        fused_output = torch.cat(weighted_features, dim=1)

        return fused_output

# Image Captioning Model with Attention Map Handling
class MedicalReportGenerator(nn.Module):
    def __init__(self, config, tokenizer):
        super(MedicalReportGenerator, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        # Initialize the image & text encoders
        if (config.enc_type == 'cnn'):
            self.prior_vision_encoder = CNNEncoder(config)
            self.prior_language_encoder = TransformerEncoder(config)
            self.current_vision_encoder = CNNEncoder(config)
            self.current_language_encoder = TransformerEncoder(config)
            self.language_encoder = TransformerEncoder(config)

        elif (config.enc_type == 'vit'):
            # Initialize the vision encoder (RAD-DINO)
            self.vision_encoder = VisionEncoder(config)
            # Initialize the language encoder (CXR-BERT)
            self.language_encoder = LanguageEncoder(config, tokenizer)

            # Initialize the LaVision & Language Adaptors
            self.prior_vision_adaptor = Adaptor(config)
            self.curr_vision_adaptor = Adaptor(config)

            self.prior_language_adaptor = Adaptor(config)
            self.curr_language_adaptor = Adaptor(config)
            self.rep_language_adaptor = Adaptor(config)

        # Initialize the Layer Normnalization
        # self.layer_norm = nn.LayerNorm(config.embed_dim)

        # Initialize TemporalFactorizer: Prior + Current Image & Text -> time shared, time specific current & prior embeddings 
        self.t_fact = TemoporalFactorizer(config)

        # Initialize Multi-Modal Fusion Network -> Prior, Current Images + Prior Report + Current Indication
        self.mmf_net = MultiModalFusionNetwork(config)

        # Initialize MoE Fusion Network
        self.moe_fusion = MoEFusionNetwork(config, num_experts=3)

        # Initialize the Language decoder
        self.decoder = LanguageDecoder(config)
        # self.decoder = LLaMAXDecoder(config)

    def forward(self, prior_images, prior_indication_tokens, prior_report_tokens, current_images, curr_indication_tokens, current_report_tokens, temperature=1.0):

        #### Embeddings from Enocder Models ####
        if self.config.enc_type == 'cnn':
            v_p = self.prior_vision_encoder(prior_images)
            v_c = self.current_vision_encoder(current_images)

            l_p = self.prior_language_encoder(prior_indication_tokens)
            l_c = self.current_language_encoder(curr_indication_tokens)

            r_p = self.language_encoder(prior_report_tokens)

        elif self.config.enc_type == 'vit':
            v_p = self.vision_encoder(prior_images)
            v_p = self.prior_vision_adaptor(v_p)

            v_c = self.vision_encoder(current_images)
            v_c = self.curr_vision_adaptor(v_c)

            l_p = self.language_encoder(prior_indication_tokens)
            l_p = self.prior_language_adaptor(l_p)

            l_c = self.language_encoder(curr_indication_tokens)
            l_c = self.curr_language_adaptor(l_c)

            r_p = self.language_encoder(prior_report_tokens)
            r_p = self.rep_language_adaptor(r_p)
    
        #### Temporal Factorization
        v_sh_c, v_sh_p, v_sp_c, v_sp_p = self.t_fact(v_c, v_p)
        l_sh_c, l_sh_p, l_sp_c, l_sp_p = self.t_fact(l_c, l_p)

        # Multi-modal Fusion
        mmfo_sp_v2l = self.mmf_net(torch.cat((v_sp_c, v_sp_p), dim=1), torch.cat((l_sp_c, l_sp_p), dim=1))
        mmfo_sp_l2v = self.mmf_net(torch.cat((l_sp_c, l_sp_p), dim=1), torch.cat((v_sp_c, v_sp_p), dim=1))
        mmfo_sp = torch.cat((mmfo_sp_v2l, mmfo_sp_l2v), dim=1)

        mmfo_sh_v2l = self.mmf_net(torch.cat((v_sh_c, v_sh_p), dim=1), torch.cat((l_sh_c, l_sh_p), dim=1))
        mmfo_sh_l2v = self.mmf_net(torch.cat((l_sh_c, l_sh_p), dim=1), torch.cat((v_sh_c, v_sh_p), dim=1))
        mmfo_sh = torch.cat((mmfo_sh_v2l, mmfo_sh_l2v), dim=1)
        
        # Multi-modal Fusion
        # mmfo = torch.cat((mmfo_sh, mmfo_sp, mmfo_vl), dim=1)

        # MoE Fusion
        mmfo = self.moe_fusion(mmfo_sh, mmfo_sp, r_p)
        
        # Language Decoder
        # logits = self.decoder(current_report_tokens, mmfo, start_pos=0, temperature=temperature)
        logits = self.decoder(current_report_tokens, mmfo, start_pos=0)

        return logits, v_sh_c, v_sh_p, v_sp_c, v_sp_p, l_sh_c, l_sh_p, l_sp_c, l_sp_p, mmfo