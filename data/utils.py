import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Any
import logging # Import the logging library

# --- Configuration ---
# Set random seed for reproducibility 
torch.manual_seed(42)

# --- Type Aliases ---
ConfigType = Any # Assume config object can provide a logger via get_logger
TokenizerType = Any
TransformType = Any 
from data import LongitudinalMIMICCXRDataset # Keep original import - Assume this defines LongitudinalDatasetType
LongitudinalDatasetType = Dataset # Replace with LongitudinalMIMICCXRDataset if imported

# --- Data Splitting ---

def train_val_test_split(
    config: ConfigType, 
    data: Any  # Still unused parameter
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset loaded via config into train, validation, and test sets.

    Assumes the DataFrame loaded by config.data_prep() has a 'split' column 
    containing 'train', 'validate', and 'test' identifiers. Uses logging for output.

    Args:
        config: A configuration object expected to have:
                - `data_prep` method returning a pandas DataFrame with a 'split' column.
                - `get_logger` method that accepts a name (str) and returns a logger instance.
        data: An argument that is currently passed but not utilized within
              this function's logic. The data is loaded via config.data_prep().

    Returns:
        A tuple containing three pandas DataFrames:
        - train_data: DataFrame containing the training set split.
        - val_data: DataFrame containing the validation set split.
        - test_data: DataFrame containing the test set split.
        
    Raises:
        AttributeError: If `config` does not have `data_prep` or `get_logger` methods.
        KeyError: If the DataFrame returned by `config.data_prep()` does not 
                  contain a 'split' column.
        TypeError: If `config.data_prep()` does not return a pandas DataFrame.
        Exception: Propagates exceptions from `config.data_prep()` or `config.get_logger()`.
    """
    # --- Basic Configuration and Logger Setup ---
    if not hasattr(config, 'get_logger'):
        raise AttributeError("config object must have a 'get_logger' method.")
    logger = config.get_logger(__name__) # Get logger instance

    logger.info("Splitting data into train, validation, and test sets...") 
    
    # Load data using the configuration object's method
    try:
        mimic_dataset: pd.DataFrame = config.data_prep() 
    except Exception as e:
        logger.error(f"Failed during config.data_prep(): {e}")
        raise

    if not isinstance(mimic_dataset, pd.DataFrame):
        error_msg = "config.data_prep() must return a pandas DataFrame."
        logger.error(error_msg)
        raise TypeError(error_msg)
        
    if 'current_split' not in mimic_dataset.columns:
        error_msg = "DataFrame from config.data_prep() is missing the required 'split' column."
        logger.error(error_msg)
        raise KeyError(error_msg)

    # Filter DataFrame based on the 'split' column
    train_data = mimic_dataset[mimic_dataset['current_split'] == 'train']
    val_data = mimic_dataset[mimic_dataset['current_split'] == 'validate']
    test_data = mimic_dataset[mimic_dataset['current_split'] == 'test']
    
    logger.info(f"Data split complete. Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")
    
    return train_data, val_data, test_data

# --- Dataset Loading ---

def load_torch_dataset(
    config: ConfigType, 
    tokenizer: TokenizerType, 
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame
) -> Tuple[LongitudinalDatasetType, LongitudinalDatasetType, LongitudinalDatasetType]:
    """
    Creates PyTorch Dataset instances for train, validation, and test splits.

    Uses the custom LongitudinalMIMICCXRDataset class and applies transformations 
    obtained from the configuration object. Uses logging for output.

    Args:
        config: A configuration object expected to have:
                - `get_image_transforms` method returning image transformations.
                - `get_logger` method that accepts a name (str) and returns a logger instance.
                - Other attributes required by LongitudinalMIMICCXRDataset.
        tokenizer: The tokenizer to be used for text processing within the dataset.
        train_data: DataFrame containing the training data.
        val_data: DataFrame containing the validation data.
        test_data: DataFrame containing the test data.

    Returns:
        A tuple containing three PyTorch Dataset objects:
        - train_dataset: Dataset for the training set.
        - val_dataset: Dataset for the validation set.
        - test_dataset: Dataset for the test set.
        
    Raises:
        AttributeError: If `config` does not have `get_image_transforms`, `get_logger`, or
                      other attributes needed by LongitudinalMIMICCXRDataset.
        Exception: Propagates exceptions from underlying operations.
    """
    # Initialize logger
    try:
        logger = config.get_logger(__name__)
    except AttributeError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.warning("config.get_logger method not found. Using basic logging configuration.")
        # Or raise AttributeError("Config object must have a 'get_logger' method.")
    except Exception as e:
        logging.error(f"Failed to get logger from config: {e}")
        raise

    logger.info("Creating PyTorch Datasets...") 
    
    # Get image transformations from the configuration
    try:
        transform: TransformType = config.get_image_transforms()
    except AttributeError as e:
        logger.error(f"Missing 'get_image_transforms' method in config: {e}")
        raise
    except Exception as e:
        logger.error(f"Error calling config.get_image_transforms(): {e}")
        raise
        
    # Instantiate Dataset objects
    try:
        train_dataset = LongitudinalMIMICCXRDataset(cfg=config, tokenizer=tokenizer, data=train_data, transform=transform)
        val_dataset = LongitudinalMIMICCXRDataset(cfg=config, tokenizer=tokenizer, data=val_data, transform=transform)
        test_dataset = LongitudinalMIMICCXRDataset(cfg=config, tokenizer=tokenizer, data=test_data, transform=transform)
    except Exception as e:
        logger.error(f"Error instantiating LongitudinalMIMICCXRDataset: {e}")
        raise # Re-raise after logging
        
    logger.info("PyTorch Datasets created successfully.")
    return train_dataset, val_dataset, test_dataset


# --- DataLoader Loading ---

def load_torch_dataloaders(
    config: ConfigType, 
    train_dataset: LongitudinalDatasetType, 
    val_dataset: LongitudinalDatasetType, 
    test_dataset: LongitudinalDatasetType
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoader instances for train, validation, and test datasets.

    Configures DataLoaders with batch size, shuffling (for train set), number of 
    workers, and pin memory settings based on the configuration and best practices.
    Uses logging for output.

    Args:
        config: A configuration object expected to have:
                - `batch_size` attribute.
                - `num_workers` attribute (optional, defaults to 4).
                - `get_logger` method that accepts a name (str) and returns a logger instance.
        train_dataset: The PyTorch Dataset for the training set.
        val_dataset: The PyTorch Dataset for the validation set.
        test_dataset: The PyTorch Dataset for the test set.

    Returns:
        A tuple containing three PyTorch DataLoader objects:
        - train_loader: DataLoader for the training set.
        - val_loader: DataLoader for the validation set.
        - test_loader: DataLoader for the test set.
        
    Raises:
        AttributeError: If `config` does not have `batch_size` or `get_logger` attributes/methods.
        Exception: Propagates exceptions from DataLoader instantiation.
    """
    # Initialize logger
    try:
        logger = config.get_logger(__name__)
    except AttributeError:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.warning("config.get_logger method not found. Using basic logging configuration.")
        # Or raise AttributeError("Config object must have a 'get_logger' method.")
    except Exception as e:
        logging.error(f"Failed to get logger from config: {e}")
        raise
        
    logger.info("Creating PyTorch DataLoaders...") 
    
    # Get DataLoader parameters from config, providing defaults
    try:
        batch_size = config.batch_size
        num_workers = getattr(config, 'num_workers', 4) # Use config value or default to 4
        logger.info(f"Using batch_size={batch_size}, num_workers={num_workers}")
    except AttributeError as e:
        logger.error(f"Missing required attribute in config (e.g., 'batch_size'): {e}")
        raise

    try:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,               
            num_workers=num_workers,    
            pin_memory=True             # Consider making pin_memory configurable via config as well
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,              
            num_workers=num_workers,
            pin_memory=False            # Typically False for val/test
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,              
            num_workers=num_workers,
            pin_memory=False            # Typically False for val/test
        )
    except Exception as e:
        logger.error(f"Error creating DataLoaders: {e}")
        raise # Re-raise after logging

    logger.info("PyTorch DataLoaders created successfully.")
    return train_loader, val_loader, test_loader