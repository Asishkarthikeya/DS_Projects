import re
import pandas as pd

from typing import List, Tuple, Any, Optional

def clean_text(text: Any,
               keep_punctuation: bool = True,
               custom_allowed_chars: Optional[str] = None,
               apply_mimic_cxr_specific: bool = True) -> str:
    """
    Cleans and normalizes a text string, with optional MIMIC-CXR specific rules.

    Steps:
    1. Handles non-string input (returns empty string).
    2. (Optional) Applies MIMIC-CXR specific structural cleaning:
        - Replaces newlines with spaces.
        - Collapses multiple underscores, spaces, or periods.
        - Removes common list numbering (e.g., "1. ", ". 2. ").
    3. Removes unwanted characters based on the cleaning strategy
       (keep_punctuation flag or custom_allowed_chars).
    4. Collapses multiple whitespace characters into a single space.
    5. Removes leading/trailing whitespace.
    6. Converts text to lowercase.

    Args:
        text: The input text (can be any type, handles non-strings).
        keep_punctuation: If True, retains common punctuation .,!?;:'"()-.
                          If False, removes all non-alphanumeric characters.
                          Ignored if custom_allowed_chars is set.
        custom_allowed_chars: Optionally provide a string of characters to keep,
                              overriding keep_punctuation. E.g., ".,?!".
                              Alphanumeric characters and whitespace are always kept.
        apply_mimic_cxr_specific: If True, applies the MIMIC-CXR structural
                                   cleaning steps (step 2).

    Returns:
        The cleaned and normalized text string.
    """
    if not isinstance(text, str):
        return '' # Return empty string for NaN or non-string types

    cleaned_text = text # Start with the original text

    # --- Step 2: (Optional) Apply MIMIC-CXR specific structural cleaning ---
    if apply_mimic_cxr_specific:
        # Replace newlines
        cleaned_text = cleaned_text.replace('\n', ' ')

        # Collapse multiple underscores, spaces, periods efficiently using regex
        # Replace 2 or more underscores with a single underscore
        cleaned_text = re.sub(r'_{2,}', '_', cleaned_text)
        # Replace 2 or more periods with a single period
        cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
        # Replace 2 or more spaces with a single space (done later again, but helpful here)
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)

        # Remove list numbering patterns (more robust with regex)
        # Pattern: Optional(dot or space) followed by number(s) followed by dot followed by space
        cleaned_text = re.sub(r'[.\s]?\d+\.\s', ' ', cleaned_text)


    # --- Step 3: Remove unwanted characters based on flags ---
    if custom_allowed_chars is not None:
        # Escape special regex characters within the custom set
        escaped_custom_chars = re.escape(custom_allowed_chars)
        # Keep alphanumeric, whitespace, and custom characters
        pattern = rf'[^A-Za-z0-9\s{escaped_custom_chars}]'
        # logging.debug(f"Using custom regex pattern: {pattern}")
    elif keep_punctuation:
        # Keep alphanumeric, whitespace, and common punctuation
        # Includes hyphen '-'
        pattern = r'[^A-Za-z0-9\s.,!?;:\'\"()-]'
        # logging.debug("Keeping common punctuation.")
    else:
        # Keep only alphanumeric and whitespace characters
        pattern = r'[^A-Za-z0-9\s]'
        # logging.debug("Removing all punctuation.")

    # Remove unwanted characters by replacing them with an empty string
    cleaned_text = re.sub(pattern, '', cleaned_text)

    # --- Step 4, 5, 6: Final whitespace collapse, strip, and lowercase ---
    # Use split/join for robust whitespace handling AFTER replacements
    cleaned_text = ' '.join(cleaned_text.split()).lower()

    return cleaned_text

def build_corpus(cfg: Any) -> Tuple[pd.DataFrame, str]:
    """
    Loads data, cleans specified text columns, combines them into a single corpus,
    and saves the corpus to a text file.

    Args:
        cfg: A configuration object/namespace expected to have attributes:
             - data_prep (callable): Function to load the initial DataFrame.
             - text_columns (List[str]): List of column names containing text to process.
             - corpus_file (str): Path to save the combined corpus file.
             - clean_keep_punctuation (bool, optional): Passed to clean_text. Defaults to True.
             - clean_custom_allowed_chars (str, optional): Passed to clean_text. Defaults to None.

    Returns:
        A tuple containing:
        - pd.DataFrame: The original DataFrame returned by cfg.data_prep().
        - str: The path to the saved corpus file.
        
    Raises:
        AttributeError: If required configuration attributes are missing in cfg.
        KeyError: If specified text columns are not found in the DataFrame.
        Exception: If cfg.data_prep() fails.
    """
    logging = cfg.get_logger(__name__)
    logging.info("Starting corpus building process...")

    # --- 1. Configuration Validation ---
    if not hasattr(cfg, 'data_prep') or not callable(cfg.data_prep):
        raise AttributeError("Configuration object 'cfg' must have a callable attribute 'data_prep'.")
    if not hasattr(cfg, 'text_columns') or not isinstance(cfg.text_columns, list):
        raise AttributeError("Configuration object 'cfg' must have a list attribute 'text_columns'.")
    if not hasattr(cfg, 'corpus_file') or not isinstance(cfg.corpus_file, str):
        raise AttributeError("Configuration object 'cfg' must have a string attribute 'corpus_file'.")
        
    # Optional cleaning parameters from config
    keep_punctuation = getattr(cfg, 'clean_keep_punctuation', True)
    custom_allowed_chars = getattr(cfg, 'clean_custom_allowed_chars', None)

    # --- 2. Load Data ---
    try:
        logging.info("Loading data using cfg.data_prep...")
        report_df = cfg.data_prep()
        if not isinstance(report_df, pd.DataFrame):
             raise TypeError("cfg.data_prep() must return a pandas DataFrame.")
        logging.info(f"Data loaded successfully. Shape: {report_df.shape}")
    except Exception as e:
        logging.error(f"Failed to load data using cfg.data_prep: {e}")
        raise

    # --- 3. Validate Columns ---
    missing_cols = [col for col in cfg.text_columns if col not in report_df.columns]
    if missing_cols:
        raise KeyError(f"The following specified text columns are missing in the DataFrame: {missing_cols}")

    # --- 4. Process Text Columns ---
    processed_series = []
    for col_name in cfg.text_columns:
        logging.info(f"Processing column: '{col_name}'...")
        # Apply the cleaning function to the column
        processed_col = report_df[col_name].apply(
            lambda x: clean_text(
                x, 
                keep_punctuation=keep_punctuation, 
                custom_allowed_chars=custom_allowed_chars
            )
        )
        processed_series.append(processed_col)
        logging.info(f"Finished processing column: '{col_name}'.")

    # --- 5. Combine Processed Text ---
    logging.info("Combining processed text columns...")
    # Ensure concatenation happens correctly even if there's only one column
    if len(processed_series) == 1:
        combined_text = processed_series[0]
    else:
        # Concatenate multiple Series element-wise with a space separator
        combined_text = processed_series[0]
        for i in range(1, len(processed_series)):
            combined_text = combined_text + " " + processed_series[i]
            
    # Ensure empty strings result from concatenating empty strings
    combined_text = combined_text.fillna('') # Replace potential NaNs from concatenation if all inputs were NaN/empty

    # --- 6. Save Corpus ---
    output_file = cfg.corpus_file
    try:
        logging.info(f"Saving combined corpus to: {output_file}")
        # Save as a single-column CSV without header or index.
        # Each row in the output file will be one combined text document.
        combined_text.to_csv(output_file, index=False, header=False, sep='\t', encoding='utf-8') 
        # Using tab separation might be safer if commas exist in the text, 
        # although to_csv handles quoting. Using utf-8 is good practice.
        logging.info(f"Corpus successfully saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save corpus to {output_file}: {e}")
        raise

    logging.info("Corpus building process finished.")
    return report_df, output_file