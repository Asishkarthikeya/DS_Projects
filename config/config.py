import os
import json
import torch
import random
import numpy as np
import pandas as pd
import logging

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)  # PyTorch RNG
random.seed(seed)        # Python's random module
np.random.seed(seed)     # NumPy RNG

class Config:
    def __init__(self, dataset_name):
        # General settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        # Set CUDA seeds if available
        if self.device == "cuda":
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
            torch.backends.cudnn.deterministic = True  # Ensure deterministic cuDNN algorithms
            torch.backends.cudnn.benchmark = False     # Disable cuDNN benchmarking

        # Dataset settings
        self.dataset_name = dataset_name
        self.image_dir = "/data/users4/nshaik3/Datasets/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
        self.data_csv = "/data/users4/nshaik3/Datasets/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/longitudinal_mimic-cxr-v2.0.csv"
        self.corpus_file = f"{dataset_name}_corpus.txt"
        self.text_columns = ['prior_indication', 'prior_findings', 'prior_impression', 'current_indication', 'current_findings', 'current_impression']
        self.vocab_size = 13500 #14500 #14855 #13563 #8400
        self.min_word_freq = 5

        # Model settings
        self.enc_type = 'vit'  # ['cnn', 'vit'] Default: vit
        self.cnn_model_name = 'efficientnet_b4'
        self.v_enc = 'facebook/dinov2-base' #['microsoft/swin-tiny-patch4-window7-224', 'facebook/dinov2-base']
        self.l_enc = 'microsoft/BiomedVLP-CXR-BERT-specialized'

        if(self.enc_type == 'cnn'):
            self.embed_dim = 1024 # {cnn: 1024, vit: 768}
            self.latent_dim = 1024 # {cnn: 1024, vit: 768}
            self.freeze_encoder = False
            self.image_size = (512, 512)
        elif(self.enc_type == 'vit'):
            self.embed_dim = 768 # {cnn: 1024, vit: 768}
            self.latent_dim = 768 # {cnn: 1024, vit: 768}
            self.freeze_encoder = True
            self.freeze_lang_encoder = True
            self.image_size = (224, 224)

        self.embed_dim = 768
        self.hidden_dim = 1024

        self.lf_num_layers = 4
        self.mmf_num_layers = 2
        self.fact_num_layers = 2
        self.num_heads = 8
        self.num_layers = 6
        
        self.num_kv_heads = 2
        self.max_seq_len = 64
        self.dropout = 0.1  # Dropout rate for regularization
        self.norm_eps = 1e-5
        self.latent_dim = 768

        # MoE Args
        self.num_experts = 4
        self.top_k = 2
        self.capacity_factor = 1.0
        self.auto_scale_hidden_dim = False

        self.decoding_strategy = 'greedy'  # ['greedy', 'sampling'] Default: greedy decoding
        self.temperature = 1.0  # 1.0 Recommended for medical reports
        self.top_p = 0.9  # Optional for top-p sampling

        # Training settings
        # self.freeze_encoder = True
        # self.freeze_lang_encoder = True
        self.epochs = 5
        self.batch_size = 4
        self.lr = 1e-4
        self.weight_decay = 1e-5
        self.beam_size = 5
        self.save_best_model = True
        self.lambda1 = 1.0 # For Cross-Entropy Loss
        self.lambda2 = 0.1 # For Orthogonality Loss
        self.lambda3 = 0.1 # For Mutual Information Loss
        self.lambda4 = 0.1 # For Report Alignment Loss
        self.lambda5 = 0.1 # For Invariance Loss

        # Image transformation settings
        # self.image_size = (224, 224)  # Resize images to (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Image VAE settings
        self.fmap_dims = (512, 16, 16) # (512, 16, 16) for (256, 256) & (512, 32, 32) for (512, 512)

        self.data_dir = "/data/users3/nshaik3/"
        self.exp_name = "Projects/ICML-2026/DiA-Long/Experiments/"
        self.exp_no = f"{self.enc_type}_2"
        self.output_dir = os.path.join(self.data_dir, self.exp_name, self.dataset_name, self.exp_no)
        # Create the directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_model_name = f"{self.epochs}_epochs.pth"
        self.loss_curve_name = f"{self.epochs}_epochs_loss_plot"
        self.results_file_name = f"{self.epochs}_epochs_predictions.csv"
        # Model saving settings
        self.model_save_path = self.output_dir + "/" + self.best_model_name
        self.loss_curve_save_path = self.output_dir + "/" + self.loss_curve_name
        self.results_file_save_path = self.output_dir + "/" + self.results_file_name

    def __repr__(self):
        return f"Config(device={self.device}, embed_dim={self.embed_dim}, " \
               f"hidden_dim={self.hidden_dim}, num_heads={self.num_heads}, " \
               f"num_layers={self.num_layers}, vocab_size={self.vocab_size}, " \
               f"max_seq_len={self.max_seq_len}, epochs={self.epochs}, " \
               f"batch_size={self.batch_size}, lr={self.lr}, weight_decay={self.weight_decay}, " \
               f"beam_width={self.beam_size}, dropout={self.dropout}, " \
               f"save_best_model={self.save_best_model}, " \
               f"lambda1={self.lambda1}, lambda2={self.lambda2}, lambda3={self.lambda3})"

    def _process_data(self,file_path):
        with open(file_path) as file:
            data_dict = json.load(file)
            
        paths, keywords, descs = [], [], []
        for item in data_dict:
            paths.append(list(item.keys())[0])
            keyword = list(item.values())[0]['keywords']
            keywords.append(keyword)
            descs.append(list(item.values())[0]['clinical-description'])
            
        data = pd.DataFrame({"image_path": paths, "keywords": keywords, "caption": descs})
        
        return data
        
    def data_prep(self):  
        # Read datasets
        data = pd.read_csv(self.data_csv)

        # Drop rows with NaN current_findings values
        data = data.dropna(subset=['current_findings'])
        
        return data

    def get_image_transforms(self):
        """
        Returns the transformation pipeline to be applied to images.
        Uses self.image_size, self.mean, and self.std from config.
        """
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_optimizer(self, model):
        """
        Returns the optimizer (AdamW) for the given model.
        """
        from torch.optim import AdamW
        return AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def load_checkpoint(self, model, checkpoint_path=None):
        """
        Load model checkpoint if available.
        If no path is given, default to loading from self.model_save_path.
        """
        checkpoint_path = checkpoint_path or self.model_save_path
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint loaded from {checkpoint_path}")
            return model
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

    def save_checkpoint(self, model, epoch, loss, acc):
        """
        Save model checkpoint at the specified path.
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'acc': acc,
        }, self.model_save_path)
        print(f"Checkpoint saved to {self.model_save_path}")

    """Logging utility."""
    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger."""
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

