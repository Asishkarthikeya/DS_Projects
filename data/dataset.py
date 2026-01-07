import os
import torch
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from data.build_corpus import clean_text


class LongitudinalMIMICCXRDataset(Dataset):
    """
    PyTorch Dataset for Longitudinal MIMIC-CXR Report Generation using Python logging.

    Loads data pairs: (prior image, prior report), current indication, (current image, current report).
    Handles missing images, prior reports, and indications with placeholders.
    Uses a logger obtained from the config object (cfg.get_logger).
    """
    def __init__(self, cfg, tokenizer, data: pd.DataFrame, transform=None):
        """
        Args:
            cfg (object): Configuration object with attributes like:
                          image_dir (str): Path to the directory containing images.
                          max_seq_len (int): Maximum sequence length for tokenized text.
                          image_size (tuple): Target image size (height, width).
                          get_logger (callable): A function that takes a name (str)
                                                 and returns a logging.Logger instance.
            tokenizer: Tokenizer instance. Must have:
                       - encode(text) -> list[int] method.
                       - pad_token_id (int) attribute or equivalent mechanism.
            data (pd.DataFrame): DataFrame containing the dataset rows for the specific split.
            transform (callable, optional): Optional transform to be applied on loaded PIL images.
        """
        super().__init__()

        # --- Basic Configuration and Logger Setup ---
        if not hasattr(cfg, 'get_logger'):
             raise AttributeError("cfg object must have a 'get_logger' method.")
        self.logger = cfg.get_logger(__name__) # Get logger instance

        if not hasattr(cfg, 'image_dir') or not hasattr(cfg, 'max_seq_len') or not hasattr(cfg, 'image_size'):
             self.logger.error("cfg object missing required attributes: 'image_dir', 'max_seq_len', or 'image_size'.")
             raise AttributeError("cfg object must contain 'image_dir', 'max_seq_len', and 'image_size' attributes.")
        if not hasattr(tokenizer, 'encode') or not (hasattr(tokenizer, 'pad_token_id') or (hasattr(tokenizer, 'word2idx') and '<pad>' in tokenizer.word2idx)):
             self.logger.error("Tokenizer missing required 'encode' method or padding mechanism.")
             raise AttributeError("tokenizer must have an 'encode' method and a 'pad_token_id' attribute (or compatible padding mechanism like word2idx['<pad>']).")

        self.cfg = cfg
        self.tokenizer = tokenizer
        # Ensure data has a contiguous index for reliable iloc access
        self.data = data.reset_index(drop=True)
        self.transform = transform
        self.image_dir = cfg.image_dir
        self.max_seq_len = cfg.max_seq_len

        # --- Determine Padding ID ---
        try:
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                self.pad_token_id = tokenizer.pad_token_id
            elif hasattr(tokenizer, 'word2idx') and '<pad>' in tokenizer.word2idx:
                self.pad_token_id = tokenizer.word2idx['<pad>']
                self.logger.warning("Using tokenizer.word2idx['<pad>'] for padding.")
            elif hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
                 self.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                 self.logger.warning("Derived pad_token_id from tokenizer.pad_token.")
            else:
                 raise ValueError("Padding token ID could not be determined.")
        except (AttributeError, TypeError, KeyError, ValueError) as e:
             self.logger.error(f"Failed to determine padding token ID: {e}")
             raise ValueError(f"Could not determine padding token ID from tokenizer: {e}")


        # --- Define Placeholders ---
        # Define placeholder text constants
        self.no_indication_text = "[NHI]" # No Header (Indication)
        self.no_prior_report_text = "[NHPR]" # No Prior Report

        # --- Pre-validate necessary columns exist ---
        required_cols = ['current_anchor_image', 'current_indication', 'current_findings', 'prior_anchor_image', 
                        'prior_indication', 'prior_findings']

        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            self.logger.error(f"Required columns missing in DataFrame: {missing_cols}")
            raise ValueError(f"Required columns missing in DataFrame: {missing_cols}")

        self.logger.info(f"Initialized {self.__class__.__name__} with {len(self.data)} samples.")
        self.logger.info(f"Image directory: {self.image_dir}")
        self.logger.info(f"Max sequence length: {self.max_seq_len}")
        self.logger.info(f"Padding token ID: {self.pad_token_id}")


    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data)
    
    def _is_valid(self, relative_path: str) -> bool:
        # Check if path is valid & return
        return not (pd.isna(relative_path) or not isinstance(relative_path, str) or not relative_path.strip())


    def _load_image(self, relative_path: str, context_idx: int) -> torch.Tensor:
        """
        Loads, transforms, and returns an image tensor.
        Returns a blank image tensor if path is invalid, image is not found,
        or loading/transform fails. Logs warnings on failure.

        Args:
            relative_path (str): The relative path from the image directory.
            context_idx (int): The dataset index for context in logging messages.

        Returns:
            torch.Tensor: The loaded and transformed image or a blank placeholder.
        """

        full_path = os.path.join(self.image_dir, relative_path)

        # 1. Try loading the image
        try:
            image = Image.open(full_path).convert("RGB")
        except FileNotFoundError:
             self.logger.warning(f"Idx {context_idx}: Image file not found: {full_path}. Returning blank.")
        except (UnidentifiedImageError, Exception) as e:
             self.logger.warning(f"Idx {context_idx}: Could not load image {full_path} due to error: {e}. Returning blank.")

        # 2. Apply transform if available
        if self.transform:
            try:
                image = self.transform(image)
                # Basic check: ensure transform returns a tensor
                if not isinstance(image, torch.Tensor):
                     self.logger.warning(f"Idx {context_idx}: Transform did not return a Tensor for {full_path}. Returning blank image.")
            except Exception as e:
                self.logger.warning(f"Idx {context_idx}: Failed to apply transform to {full_path}: {e}. Returning blank.")
        else:
            # If no transform, we cannot guarantee a tensor output
            self.logger.error(f"Idx {context_idx}: Image transform is required but not provided/functional for {full_path}.")
            raise RuntimeError("Image transform is required to ensure tensor output.")

        return image

    def _tokenize_and_pad(self, text: str) -> torch.Tensor:
        """Tokenizes text, truncates, and pads to max_seq_len."""
        try:
            # Use add_special_tokens=False if tokenizer adds BOS/EOS automatically
            tokens = self.tokenizer.encode(text, add_special_tokens=False)

            # Truncate if necessary
            tokens = tokens[:self.max_seq_len]

            # Pad if necessary
            padding_needed = self.max_seq_len - len(tokens)
            if padding_needed > 0:
                tokens = tokens + [self.pad_token_id] * padding_needed

            return torch.tensor(tokens, dtype=torch.long)
        except Exception as e:
            self.logger.error(f"Failed during tokenize/pad for text starting with '{text[:50]}...': {e}", exc_info=True)
             # Return tensor of pad tokens on error to avoid crashing dataloader
            return torch.full((self.max_seq_len,), self.pad_token_id, dtype=torch.long)


    def __getitem__(self, idx: int) -> tuple: #dict:
        """
        Retrieves a single data sample, including images and tokenized text.

        Returns:
            A dictionary containing:
            - 'current_image': Tensor of the transformed current image.
            - 'prior_image': Tensor of the transformed prior image (or blank placeholder).
            - 'current_indication_tokens': Tensor of tokenized/padded current indication (or placeholder).
            - 'prior_report_tokens': Tensor of tokenized/padded prior report (or placeholder).
            - 'current_report_tokens': Tensor of tokenized/padded current report (target).
            - 'id': The original study/image identifier (optional, for reference).
        """
        # 1. Get the row data using iloc for efficiency
        try:
            row = self.data.iloc[idx]
            # Get an identifier for logging context, default to index
            item_id = row.get('id', f"index_{idx}")
        except IndexError:
            self.logger.error(f"Index {idx} out of bounds for dataset with length {len(self)}.")
            # Re-raising might be better depending on how DataLoader handles it
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self)}.")

        # 2. Load Images (pass index/id for context)
        current_image = self._load_image(row['current_anchor_image'], item_id)

        if self._is_valid(row['prior_anchor_image']):
            prior_image = self._load_image(row['prior_anchor_image'], item_id)
        else:
            prior_image = self._load_image(row['current_anchor_image'], item_id)
            

        # 3. Process Text (Handle missing, clean, tokenize, pad)
        # Current Indication
        curr_indication_text = row.get('current_indication', None)
        curr_indication_text = clean_text(curr_indication_text)
        if pd.isna(curr_indication_text) or not isinstance(curr_indication_text, str) or not curr_indication_text.strip():
            curr_indication_text = self.no_indication_text
            # self.logger.debug(f"Item {item_id}: Using placeholder '{self.no_indication_text}' for missing indication.")
        curr_indication_tokens = self._tokenize_and_pad(curr_indication_text)

        # Prior Indication
        prior_indication_text = row.get('prior_indication', None)
        prior_indication_text = clean_text(prior_indication_text)
        if pd.isna(prior_indication_text) or not isinstance(prior_indication_text, str) or not prior_indication_text.strip():
            prior_indication_text = self.no_indication_text
            # self.logger.debug(f"Item {item_id}: Using placeholder '{self.no_indication_text}' for missing indication.")
        prior_indication_tokens = self._tokenize_and_pad(prior_indication_text)

        # Prior Findings (Report)
        prior_text = row.get('prior_findings', None)
        prior_text = clean_text(prior_text)
        if pd.isna(prior_text) or not isinstance(prior_text, str) or not prior_text.strip():
            prior_text = self.no_prior_report_text
            self.logger.debug(f"Item {item_id}: Using placeholder '{self.no_prior_report_text}' for missing prior report.")
        prior_report_tokens = self._tokenize_and_pad(prior_text)

        # Current Findings (Report) - Target
        current_text = row.get('current_findings', None)
        current_text = clean_text(current_text)
        if pd.isna(current_text) or not isinstance(current_text, str) or not current_text.strip():
             self.logger.warning(f"Item {item_id}: Missing target 'findings'. Resulting sequence will be padding tokens.")
             current_text = "" # Results in only padding tokens
        current_report_tokens = self._tokenize_and_pad(current_text)

        return prior_image, prior_indication_tokens, prior_report_tokens, current_image, curr_indication_tokens, current_report_tokens