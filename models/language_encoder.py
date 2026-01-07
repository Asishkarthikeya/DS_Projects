#language_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config

class LanguageEncoder(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.model_name = config.l_enc
        self.tokenizer = tokenizer
        self.logger = config.get_logger(__name__)
        self.logger.info(f"Initializing Language Encoder from: {self.model_name}")
        self.freeze_lang_encoder = config.freeze_lang_encoder
        try:
            # Load the pre-trained BERT-like model
            self.encoder = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

            # Verify dimension
            if hasattr(self.encoder.config, 'hidden_size'):
                 model_dim = self.encoder.config.hidden_size
                 if model_dim != config.embed_dim:
                     self.logger.warning(f"CXR-BERT model hidden size ({model_dim}) != config TEXT_FEATURE_DIM ({config.embed_dim}). Check config.")
            else:
                 self.logger.warning("Could not automatically determine CXR-BERT output dimension. Assuming config.embed_dim is correct.")

            # We also need the tokenizer associated with this model for potential internal use
            # Although tokenization happens in the Dataset, having it here can be useful.
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        except Exception as e:
            self.logger.error(f"Failed to load Language Encoder model '{self.model_name}': {e}")
            raise

        if self.freeze_lang_encoder:
            self.logger.info("Freezing Language Encoder parameters.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
             self.logger.info("Text Encoder parameters are trainable.")


    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Forward pass through the text encoder.

        Args:
            input_ids (torch.Tensor): Batch of token IDs, shape (B, SeqLen).
            attention_mask (torch.Tensor): Batch of attention masks, shape (B, SeqLen).
            token_type_ids (torch.Tensor, optional): Batch of token type IDs, shape (B, SeqLen).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - last_hidden_state: Token embeddings (B, SeqLen, FeatureDim).
                - pooler_output: Usually the [CLS] token embedding transformed (B, FeatureDim).
        """

        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state # (B, t, d) in paper notation

        return last_hidden_state

    def train(self, mode=True):
        super().train(mode)
        # Keep the underlying transformer in eval mode if it's frozen
        if hasattr(self, 'encoder') and self.encoder.parameters().__next__().requires_grad == False:
             self.encoder.eval()