#### Compute Metrics ####
import nltk
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
#!pip install rouge
from rouge import Rouge
#!pip install bert_score
from bert_score import score
import matplotlib.pyplot as plt
from training.evaluate import evaluate_model
# Initialize BLEU
nltk.download('punkt')

def compute_metrics(references, hypotheses, actual_predicted_samples):

    # Tokenize the captions
    actual_tokens = [tokens.split() for tokens in references]
    predicted_tokens = [tokens.split() for tokens in hypotheses]

    # Create reference list for corpus_bleu (list of lists of references)
    reference_tokens = [[actual] for actual in actual_tokens]

    # Compute BLEU scores
    b1 = corpus_bleu(reference_tokens, predicted_tokens, weights=(1.0, 0, 0, 0))
    b2 = corpus_bleu(reference_tokens, predicted_tokens, weights=(0.5, 0.5, 0, 0))
    b3 = corpus_bleu(reference_tokens, predicted_tokens, weights=(0.33, 0.33, 0.33, 0))
    b4 = corpus_bleu(reference_tokens, predicted_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    # ROUGE score
    rouge_scorer = Rouge()
    rouge_scores = rouge_scorer.get_scores(hypotheses, references, avg=True)
    rouge_score = rouge_scores['rouge-1']['f']

    # Compute BERTScore
    reference_captions = [tokens for tokens in references]
    prediction_captions = [tokens for tokens in hypotheses]
    P, R, F1 = score(prediction_captions, reference_captions, lang="en", verbose=False)

    # Display BLEU scores
    print('BLEU-1: %f' % b1)
    print('BLEU-2: %f' % b2)
    print('BLEU-3: %f' % b3)
    print('BLEU-4: %f' % b4)
    print('BLEU Avg: %f' % ((b1 + b2 + b3 + b4) / 4))

    print("ROUGE Score:", rouge_score)

    # Display average BERTScore for Precision, Recall, and F1
    print("BERTScore Precision:", P.mean().item())
    print("BERTScore Recall:", R.mean().item())
    print("BERTScore F1:", F1.mean().item())

    # Display 5 sample captions with images
    print("\nSample Captions:")
    for sample in actual_predicted_samples[:5]:
        # Display the image along with the actual and predicted captions
        print(f"Prior Indication: {sample['prior_indocation']}\nPrior Report: {sample['prior_report']}\nCurrent Indicationt: {sample['current_indication']}\nActual Report: {sample['actual_caption']}\nGenerated Report: {sample['predicted_caption']}\n\n")
        # plt.figure(figsize=(5, 5))
        # plt.imshow(sample["image"])
        # plt.axis('off')
        #plt.title()
        # plt.show()