import torch
import sys
from config import Config
from data import Tokenizer
from data.build_corpus import build_corpus
from data.utils import train_val_test_split, load_torch_dataloaders, load_torch_dataset
from models import MedicalReportGenerator
from training.train import train
from utils.compute_params import compute_parameters
from training.metrics import compute_metrics
from training.evaluate import evaluate_model
from utils.write_to_csv import write_to_csv
from utils.compute_flops import compute_flops
# Set the seed for reproducibility
torch.manual_seed(42)

def main():
    # Initialize model, criterion, optimizer, and dataloaders
    dataset_name = sys.argv[1]
    print("Loading the configuration for dataset: ", dataset_name)
    config = Config(dataset_name)
    print("Using Encoder Type: ", config.enc_type)

    # Build the corpus file
    report_df, output_file = build_corpus(config)
    tokenizer = Tokenizer(corpus_file=output_file, vocab_size=config.vocab_size, min_freq=config.min_word_freq)

    # Get the train_test_data
    train_data, val_data, test_data = train_val_test_split(config, report_df)

    # Load the pytorch dataset
    train_dataset, val_dataset, test_dataset = load_torch_dataset(config, tokenizer, train_data, val_data, test_data)
    # train_dataset, val_dataset, test_dataset = load_torch_dataset(config, tokenizer, train_data[:70], val_data[:10], test_data[:20]) # debugging with few samples

    # Initialize the dataloaders
    train_loader, val_loader, test_loader = load_torch_dataloaders(config, train_dataset, val_dataset, test_dataset)

    # Train the model
    model = MedicalReportGenerator(config, tokenizer).to(config.device)
    train(model, train_loader, val_loader, test_loader, config, tokenizer)

    # Load the best model from checkpoint
    config.load_checkpoint(model)

    # Compute parameters
    trainable, non_trainable, total = compute_parameters(model)
    print(f"Trainable Parameters: {trainable}")
    print(f"Non-Trainable Parameters: {non_trainable}")
    print(f"Total Parameters: {total}")

    references, hypotheses, actual_predicted_samples = evaluate_model(model, config, test_loader)

    write_to_csv(actual_predicted_samples, config.results_file_save_path)
    
    compute_metrics(references, hypotheses, actual_predicted_samples)

    flops = compute_flops(model, config)
    print("FLOPs: ", flops / 1e9, "GFLOPs")


if __name__ == "__main__":
    main()