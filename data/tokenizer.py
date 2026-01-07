import logging
from collections import Counter
from itertools import chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tokenizer:
    """
    Simple vocabulary and tokenizer based on word frequency.

    Builds a vocabulary from a corpus file, filtering infrequent words
    and reserving indices for special tokens including <pad>, <start>, <end>,
    <unk>, [NHI], and [NHPR].
    """
    def __init__(self, corpus_file, vocab_size=3000, min_freq=3):
        """
        Initializes the Tokenizer.

        Args:
            corpus_file (str): Path to the text file containing captions/reports,
                               one per line.
            vocab_size (int): The target total size of the vocabulary, including
                              all special tokens.
            min_freq (int): The minimum frequency for a word to be included
                            in the vocabulary (before applying vocab_size limit).
        """
        logger.info(f"Initializing Tokenizer from corpus: {corpus_file}")
        logger.info(f"Target vocabulary size: {vocab_size}, Minimum word frequency: {min_freq}")

        # --- Define Special Tokens ---
        # Keep track of them for easier management
        self.special_tokens = {
            "<pad>": 0,  # Padding
            "<start>": 1,  # Start of sequence
            "<end>": 2,  # End of sequence
            "<unk>": 3,  # Unknown word
            "[NHI]": 4,  # No Header / Indication placeholder
            "[NHPR]": 5,  # No Prior Report placeholder
        }
        self.num_special_tokens = len(self.special_tokens)
        self.pad_token_id = self.special_tokens["<pad>"]
        self.unk_token_id = self.special_tokens["<unk>"]
        self.start_token_id = self.special_tokens["<start>"]
        self.end_token_id = self.special_tokens["<end>"]

        # Ensure vocab_size is large enough for special tokens
        if vocab_size < self.num_special_tokens:
            raise ValueError(f"vocab_size ({vocab_size}) must be at least {self.num_special_tokens} "
                             f"to accommodate all special tokens.")

        # --- Build Vocabulary ---
        try:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                # Read lines, strip whitespace, handle potential empty lines
                captions = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            logger.error(f"Corpus file not found: {corpus_file}")
            raise
        except Exception as e:
            logger.error(f"Error reading corpus file {corpus_file}: {e}")
            raise

        if not captions:
            logger.warning(f"Corpus file {corpus_file} is empty or contains only whitespace.")
            # Initialize with only special tokens if corpus is empty
            self.word2idx = self.special_tokens.copy()
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
            self.vocab_size = self.num_special_tokens
            logger.info(f"Initialized tokenizer with {self.vocab_size} special tokens only (empty corpus).")
            return

        # Calculate word frequency
        logger.info("Calculating word frequencies...")
        word_freq = Counter(chain.from_iterable(caption.split() for caption in captions))
        logger.info(f"Found {len(word_freq)} unique words initially.")

        # Filter words by minimum frequency
        filtered_word_freq = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
        logger.info(f"Kept {len(filtered_word_freq)} words after filtering (min_freq={min_freq}).")

        # Calculate how many words to keep from the corpus based on target vocab size
        num_words_to_keep = vocab_size - self.num_special_tokens

        # Get the most common words from the filtered list
        most_common = Counter(filtered_word_freq).most_common(num_words_to_keep)
        logger.info(f"Selected {len(most_common)} most common words to fit target vocab size.")

        # --- Create Mappings ---
        # Start indexing corpus words *after* the special tokens
        self.word2idx = {word: i + self.num_special_tokens for i, (word, _) in enumerate(most_common)}
        # Add the special tokens with their predefined indices
        self.word2idx.update(self.special_tokens)

        # Create the reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # Store the actual final vocabulary size
        self.vocab_size = len(self.word2idx)
        logger.info(f"Final vocabulary size: {self.vocab_size}")
        if self.vocab_size < vocab_size:
             logger.warning(f"Actual vocabulary size ({self.vocab_size}) is less than target ({vocab_size}) "
                            f"due to filtering and available words.")


    def encode(self, text: str, add_special_tokens=True) -> list[int]:
        """
        Converts a text string into a list of token indices.

        Args:
            text (str): The input text string.
            add_special_tokens (bool): Whether to add <start> and <end> tokens.
                                       Defaults to True.

        Returns:
            list[int]: A list of token indices corresponding to the text.
        """
        if not isinstance(text, str):
             logger.warning(f"Input to encode is not a string: {type(text)}. Returning empty list.")
             return []
        tokens = text.split()
        if add_special_tokens:
            tokens = ["<start>"] + tokens + ["<end>"]
        # Use .get() with default pointing to <unk> index
        return [self.word2idx.get(word, self.unk_token_id) for word in tokens]

    def decode(self, indices: list[int], skip_special_tokens=True) -> str:
        """
        Converts a list of token indices back into a text string.

        Args:
            indices (list[int]): The list of token indices.
            skip_special_tokens (bool): Whether to skip all special tokens
                                        (indices 0-5 in this case) during decoding.
                                        Defaults to True.

        Returns:
            str: The decoded text string.
        """
        if not isinstance(indices, (list, tuple)):
            logger.warning(f"Input to decode is not a list or tuple: {type(indices)}. Returning empty string.")
            return ""

        if skip_special_tokens:
            # Skip tokens with indices less than num_special_tokens
            return " ".join(self.idx2word.get(idx, "<unk>") # Use .get for robustness
                            for idx in indices if idx >= self.num_special_tokens)
        else:
            return " ".join(self.idx2word.get(idx, "<unk>") for idx in indices)

    def __len__(self):
        """Returns the size of the vocabulary."""
        return self.vocab_size