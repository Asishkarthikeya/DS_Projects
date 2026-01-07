import os
import time 
import torch
import torch.nn as nn
from tqdm import tqdm
from training.metrics import compute_metrics
from training.evaluate import evaluate_model
from utils.visualization import plot_loss_curves
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from training.losses import RKDLoss, OrthogoDiffLoss, ContrastiveAlignmentLoss, SharedInformationAlignmentLoss, HSIC

def train(model, train_loader, val_loader, test_loader, config, tokenizer):

    """Training loop for the MedicalReportGenerator model."""
    logger = config.get_logger(__name__) # Get logger instance

    # Track loss history
    train_losses, val_losses = [], []

    # Optimizer and criterion
    optimizer = config.get_optimizer(model)
    decoder_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<pad>"])


    rkd_criterion = RKDLoss()
    orth_criterion = OrthogoDiffLoss()
    mi_criterion = SharedInformationAlignmentLoss(temperature=0.1)
    contrastive_criterion = ContrastiveAlignmentLoss(temperature=0.1)
    hsic_criterion = HSIC(sigma=1.0)

    # --- Hyperparameters for new loss terms ---
    alpha_rkdl = config.lambda1
    beta_orth = config.lambda2
    gamma_mi = config.lambda3
    lambda_align = config.lambda4
    delta_inv = config.lambda5

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Mixed precision scaler
    scaler = GradScaler()

    # Helper function to compute accuracy during validation
    def compute_accuracy(outputs, captions):
        # Get the predicted tokens (argmax of logits)
        predicted = outputs.argmax(dim=-1)
        # Exclude <pad> tokens and calculate accuracy
        correct = (predicted == captions).float()
        correct = correct[captions != tokenizer.word2idx["<pad>"]]  # Ignore padding tokens
        accuracy = correct.sum() / correct.size(0)  # Accuracy as percentage
        return accuracy.item()

    # Training and validation loop
    best_acc = float(0.0)
    best_loss = float('inf')

    # Record training_start_time
    training_start_time = time.time()
    for epoch in range(config.epochs):
        
        # Record train epoch start time
        train_epoch_start_time = time.time()
        
        model.train()
        train_loss = 0

        # TQDM for training
        with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{config.epochs}") as pbar_train:
            for prior_images, prior_indication_tokens, prior_report_tokens, current_images, curr_indication_tokens, current_report_tokens in train_loader:
                prior_images, prior_indication_tokens, prior_report_tokens, current_images, curr_indication_tokens, current_report_tokens = prior_images.to(config.device), prior_indication_tokens.to(config.device), prior_report_tokens.to(config.device), current_images.to(config.device), curr_indication_tokens.to(config.device), current_report_tokens.to(config.device)

                # Forward pass with mixed precision
                optimizer.zero_grad()
                with autocast():
                    outputs, v_sh_c, v_sh_p, v_sp_c, v_sp_p, l_sh_c, l_sh_p, l_sp_c, l_sp_p, mmfo = model(
                        prior_images, prior_indication_tokens, prior_report_tokens,
                        current_images, curr_indication_tokens, current_report_tokens[:, :-1],
                        temperature=1.0 # temperature set to 1.0 during training to avoid logit scaling
                    )  # Exclude <end> token and pass temperature
                    # Loss Terms
                    decoder_loss = decoder_criterion(outputs.reshape(-1, config.vocab_size), current_report_tokens[:, 1:].reshape(-1))  # Shift target by 1

                    # --- Structured Alignment Loss ---
                    # --- RKD Loss ---
                    v_feats = torch.mean(torch.cat([(v_sh_c + v_sh_p), v_sp_c, v_sp_p], dim=1), dim=1)  # [B, 768]
                    l_feats = torch.mean(torch.cat([(l_sh_c + l_sh_p), l_sp_c, l_sp_p], dim=1), dim=1)  # [B, 768]
                    rkdl = rkd_criterion(v_feats, l_feats)

                    # --- Disentanglement and Alignment Losses ---
                    # --- Compute shared features for vision and language ---
                    v_sh = torch.mean(v_sh_c + v_sh_p, dim=1)
                    l_sh = torch.mean(l_sh_c + l_sh_p, dim=1)
                    # --- Pool vision features over sequence ---
                    v_sp_p = torch.mean(v_sp_p, dim=1)
                    v_sp_c = torch.mean(v_sp_c, dim=1)
                    # --- Pool language features over sequence ---
                    l_sp_p = torch.mean(l_sp_p, dim=1)
                    l_sp_c = torch.mean(l_sp_c, dim=1)

                    # # The Invariance Loss (using Mean Squared Error) >> Ensures shared features from prior and current are similar
                    # # Pool shared features to get global shared features
                    # z_sh_v_c = torch.mean(v_sh_c, dim=1)
                    # z_sh_v_p = torch.mean(v_sh_p, dim=1)
                    # z_sh_l_c = torch.mean(l_sh_c, dim=1)
                    # z_sh_l_p = torch.mean(l_sh_p, dim=1)

                    # #  Invariance Loss using MSE
                    # inv_loss_vision = F.mse_loss(z_sh_v_c, z_sh_v_p)
                    # inv_loss_language = F.mse_loss(z_sh_l_c, z_sh_l_p)
                    # total_inv_loss = inv_loss_vision + inv_loss_language

                    # # Independence Loss
                    # # Combine current specific features
                    # z_sp_c = v_sp_c + l_sp_c 
                    # # nd total_ind_loss is the sum of HSIC between shared and specific factors
                    # total_ind_loss = hsic_criterion(z_sh_v_c, z_sp_c) + hsic_criterion(z_sh_l_c, z_sp_c)

                    # --- Orthogonality Loss (Intra-Modality Disentanglement) ---
                    orth_loss_vision = orth_criterion(v_sp_p, v_sp_c) + orth_criterion(v_sp_c, v_sh) + orth_criterion(v_sh, v_sp_p)
                    orth_loss_language = orth_criterion(l_sp_p, l_sp_c) + orth_criterion(l_sp_c, l_sh) + orth_criterion(l_sh, l_sp_p)
                    total_orth_loss = orth_loss_vision + orth_loss_language

                    # --- Mutual Information Maximization (Intra-Modality Predictiveness) ---
                    mi_loss_vision = mi_criterion(v_sp_p, v_sp_c, v_sh)
                    mi_loss_language = mi_criterion(l_sp_p, l_sp_c, l_sh)
                    total_mi_loss = mi_loss_vision + mi_loss_language

                    # --- Contrastive Alignment Loss ---
                    # Get ground-truth report embeddings using the language encoder
                    # We use the full report here, not the shifted version
                    gt_report_embs = model.language_encoder(current_report_tokens)
                    
                    # Pool token-level features to get global vectors
                    visual_global = torch.mean(mmfo, dim=1) # (B, D)
                    text_global = torch.mean(gt_report_embs, dim=1) # (B, D)
                    
                    # Calculate contrastive loss
                    align_loss = contrastive_criterion(visual_global, text_global)                    
                    
                    # Total Loss
                    loss = (decoder_loss + 
                            alpha_rkdl * rkdl +
                            beta_orth * total_orth_loss +
                            gamma_mi * total_mi_loss + 
                            lambda_align * align_loss 
                            #+ delta_inv * (total_inv_loss ) #+ total_ind_loss
                        )                  
                    
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                pbar_train.set_postfix({"Batch Loss": loss.item()})
                pbar_train.update(1)

        # Step the learning rate scheduler
        scheduler.step()

        # Record train epoch end time
        train_epoch_end_time = time.time()
        # Find total train epoch time
        train_epoch_total_time = train_epoch_end_time - train_epoch_start_time
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss, val_accuracy = 0, 0

        # Record val epoch start time
        val_epoch_start_time = time.time()
        
        # TQDM for validation
        with tqdm(total=len(val_loader), desc=f"Validating Epoch {epoch + 1}/{config.epochs}") as pbar_val:
            for prior_images, prior_indication_tokens, prior_report_tokens, current_images, curr_indication_tokens, current_report_tokens in val_loader:
                prior_images, prior_indication_tokens, prior_report_tokens, current_images, curr_indication_tokens, current_report_tokens = prior_images.to(config.device), prior_indication_tokens.to(config.device), prior_report_tokens.to(config.device), current_images.to(config.device), curr_indication_tokens.to(config.device), current_report_tokens.to(config.device)

                # Forward pass with mixed precision
                with autocast():
                    outputs, v_sh_c, v_sh_p, v_sp_c, v_sp_p, l_sh_c, l_sh_p, l_sp_c, l_sp_p, mmfo = model(
                        prior_images, prior_indication_tokens, prior_report_tokens,
                        current_images, curr_indication_tokens, current_report_tokens[:, :-1],
                        temperature=getattr(config, 'temperature', 1.0)
                    )  # Exclude <end> token and pass temperature
                    decoder_loss = decoder_criterion(outputs.reshape(-1, config.vocab_size), current_report_tokens[:, 1:].reshape(-1))  # Shift target by 1

                   
                    # --- Structured Alignment Loss ---
                    # --- RKD Loss ---
                    v_feats = torch.mean(torch.cat([(v_sh_c + v_sh_p), v_sp_c, v_sp_p], dim=1), dim=1)  # [B, 768]
                    l_feats = torch.mean(torch.cat([(l_sh_c + l_sh_p), l_sp_c, l_sp_p], dim=1), dim=1)  # [B, 768]
                    rkdl = rkd_criterion(v_feats, l_feats)

                    # --- Disentanglement and Alignment Losses ---
                    # --- Compute shared features for vision and language ---
                    v_sh = torch.mean(v_sh_c + v_sh_p, dim=1)
                    l_sh = torch.mean(l_sh_c + l_sh_p, dim=1)
                    # --- Pool vision features over sequence ---
                    v_sp_p = torch.mean(v_sp_p, dim=1)
                    v_sp_c = torch.mean(v_sp_c, dim=1)
                    # --- Pool language features over sequence ---
                    l_sp_p = torch.mean(l_sp_p, dim=1)
                    l_sp_c = torch.mean(l_sp_c, dim=1)

                    # # The Invariance Loss (using Mean Squared Error) >> Ensures shared features from prior and current are similar
                    # # Pool shared features to get global shared features
                    # z_sh_v_c = torch.mean(v_sh_c, dim=1)
                    # z_sh_v_p = torch.mean(v_sh_p, dim=1)
                    # z_sh_l_c = torch.mean(l_sh_c, dim=1)
                    # z_sh_l_p = torch.mean(l_sh_p, dim=1)

                    # #  Invariance Loss using MSE
                    # inv_loss_vision = F.mse_loss(z_sh_v_c, z_sh_v_p)
                    # inv_loss_language = F.mse_loss(z_sh_l_c, z_sh_l_p)
                    # total_inv_loss = inv_loss_vision + inv_loss_language

                    # # Independence Loss
                    # # Combine current specific features
                    # z_sp_c = v_sp_c + l_sp_c 
                    # # nd total_ind_loss is the sum of HSIC between shared and specific factors
                    # total_ind_loss = hsic_criterion(z_sh_v_c, z_sp_c) + hsic_criterion(z_sh_l_c, z_sp_c)

                    # --- Orthogonality Loss (Intra-Modality Disentanglement) ---
                    orth_loss_vision = orth_criterion(v_sp_p, v_sp_c) + orth_criterion(v_sp_c, v_sh) + orth_criterion(v_sh, v_sp_p)
                    orth_loss_language = orth_criterion(l_sp_p, l_sp_c) + orth_criterion(l_sp_c, l_sh) + orth_criterion(l_sh, l_sp_p)
                    total_orth_loss = orth_loss_vision + orth_loss_language
                    
                    # --- Mutual Information Maximization (Intra-Modality Predictiveness) ---
                    mi_loss_vision = mi_criterion(v_sp_p, v_sp_c, v_sh)
                    mi_loss_language = mi_criterion(l_sp_p, l_sp_c, l_sh)
                    total_mi_loss = mi_loss_vision + mi_loss_language

                    # --- Contrastive Alignment Loss ---
                    # Get ground-truth report embeddings using the language encoder
                    # We use the full report here, not the shifted version
                    gt_report_embs = model.language_encoder(current_report_tokens)
                    
                    # Pool token-level features to get global vectors
                    visual_global = torch.mean(mmfo, dim=1) # (B, D)
                    text_global = torch.mean(gt_report_embs, dim=1) # (B, D)
                    
                    # Calculate contrastive loss
                    align_loss = contrastive_criterion(visual_global, text_global)                    
                    
                    # Total Loss
                    loss = (decoder_loss + 
                            alpha_rkdl * rkdl +
                            beta_orth * total_orth_loss +
                            gamma_mi * total_mi_loss + 
                            lambda_align * align_loss 
                            # + delta_inv * (total_inv_loss ) #+ total_ind_loss
                        )            
                    
                if loss.isnan():
                    logger.info("NaN loss encountered during validation. Skipping this batch.")
                    continue
                else:
                    val_loss += loss.item()

                # Compute accuracy
                accuracy = compute_accuracy(outputs, current_report_tokens[:, 1:])
                val_accuracy += accuracy

                pbar_val.set_postfix({"Batch Loss": loss.item(), "Accuracy": accuracy})
                pbar_val.update(1)

        # Record val epoch end time
        val_epoch_end_time = time.time()
        # Find total val epoch time
        val_epoch_total_time = val_epoch_end_time - val_epoch_start_time
        
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)  # Average accuracy across all batches
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Train Time: {train_epoch_total_time} sec, Val Time: {val_epoch_total_time} sec")

        # Save the best model based on validation accuracy
        improved = (val_accuracy > best_acc) or (val_loss < best_loss)
        if improved and config.save_best_model:
            config.save_checkpoint(model, epoch + 1, val_loss, val_accuracy)
            best_acc = max(best_acc, val_accuracy)
            best_loss = min(best_loss, val_loss)

        # Save intermediate checkpoints every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config.output_dir, f"epoch_{epoch + 1}_checkpoint.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, checkpoint_path)
            print(f"Intermediate checkpoint saved to {checkpoint_path}")

            # Evaluate the model on the test set
            references, hypotheses, actual_predicted_samples = evaluate_model(model, config, test_loader)
            compute_metrics(references, hypotheses, actual_predicted_samples)

    # Record training_end_time
    training_end_time = time.time()

    total_time = training_end_time - training_start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Plot the loss curves
    plot_loss_curves(train_losses, val_losses, config)