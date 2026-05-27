import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
import math
import random
import threading
import time
import json
import os
from pathlib import Path
import psutil
import gc
from datetime import datetime
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters matching the research paper"""
    
    # Model Selection (Table II)
    MODEL_NAME = "microsoft/DialoGPT-medium"  # 354M params - manageable for demo
    # For actual deployment: "microsoft/phi-2" (2.7B) or "microsoft/Phi-3.5-mini-instruct" (3.8B)
    
    # Compression Parameters (Section IV)
    PRUNE_RATIO = 0.20              # 20% neuron pruning (structured)
    ALPHA = 0.6                      # Gradient vs activation weight (Eq. 1)
    LAMBDA = 1.5                     # QUBO penalty parameter (Eq. 2)
    
    # Calibration (Section IV.C)
    NUM_CALIB_SAMPLES = 100         # Reduced for faster demo, paper uses 1000
    
    # Fine-tuning (Section IV.E)
    FINE_TUNE_STEPS = 100           # Paper uses 500
    BATCH_SIZE = 2
    GRADIENT_ACCUM_STEPS = 2        # Effective batch size: 4
    LEARNING_RATE = 1e-5
    WARMUP_STEPS = 10
    
    # QUBO/Simulated Annealing (Section IV.D)
    T0 = 10.0                       # Initial temperature
    T_MIN = 0.01                    # Minimum temperature
    ALPHA_COOLING = 0.95            # Cooling rate
    SA_ITERATIONS_PER_TEMP = 100    # Iterations per temperature
    MAX_SA_ITERATIONS = 400         # Total SA iterations
    
    # Evaluation
    MAX_LENGTH = 256
    QUALITY_THRESHOLD = 0.95        # 95% quality retention target
    
    # Quantization (Section IV.F, Table III)
    QUANT_DTYPE = torch.qint8       # INT8 quantization
    
    # Output
    OUTPUT_DIR = "./compressed_model"
    METRICS_FILE = "compression_metrics.json"


# ==================== COMPRESSION PIPELINE ====================
class HybridCompressionPipeline:
    """
    Complete implementation of the hybrid compression pipeline from the paper
    Stages: Load → Importance Analysis → QUBO Pruning → Fine-tuning → Quantization
    """
    
    def __init__(self, callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.callback = callback  # For GUI updates
        self.model = None
        self.tokenizer = None
        self.quantized_model = None
        self.importance_scores = {}
        self.pruning_masks = {}
        self.metrics = OrderedDict()
        self.calib_dataset = []
        
        # Create output directory
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
        
    def log(self, message):
        """Send log message to GUI"""
        if self.callback:
            self.callback(message)
        print(message)
    
    # ==================== STAGE 1: MODEL LOADING ====================
    def load_model(self):
        """Load pre-trained model and tokenizer (Section IV.B)"""
        try:
            self.log("📦 Loading model and tokenizer...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                Config.MODEL_NAME,
                torch_dtype=torch.float32,  # FP32 for pruning
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Calculate baseline metrics
            param_count = sum(p.numel() for p in self.model.parameters())
            self.metrics['original_params'] = param_count
            self.metrics['original_size_gb'] = self._calculate_model_size(self.model)
            self.metrics['device'] = str(self.device)
            self.metrics['model_name'] = Config.MODEL_NAME
            
            result = (
                f"✅ Model loaded successfully\n"
                f"   Model: {Config.MODEL_NAME}\n"
                f"   Parameters: {param_count:,}\n"
                f"   Size: {self.metrics['original_size_gb']:.3f} GB\n"
                f"   Device: {self.device}"
            )
            
            self.log(result)
            return True, result
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    def _calculate_model_size(self, model):
        """Calculate model size in GB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024**3)
    
    # ==================== STAGE 2: CALIBRATION DATA ====================
    def prepare_calibration_data(self):
        """Prepare diverse calibration dataset (Section IV.C)"""
        try:
            self.log("📚 Preparing calibration dataset...")
            
            # Diverse prompts covering multiple domains (as per paper)
            calibration_prompts = [
                # Technical/AI
                "Large language models use transformer architectures to process sequential data efficiently.",
                "Neural network compression techniques include pruning, quantization, and knowledge distillation.",
                "The attention mechanism allows models to focus on relevant parts of the input sequence.",
                "Edge computing enables AI inference on resource-constrained mobile devices.",
                "Quantization reduces model precision from 32-bit floating point to 8-bit integers.",
                
                # Reasoning
                "If all mammals are warm-blooded and whales are mammals, then whales must be warm-blooded.",
                "The fastest way to solve this problem is to break it down into smaller subproblems.",
                "Logical reasoning requires careful analysis of premises and valid inference patterns.",
                
                # General Knowledge
                "The solar system consists of the sun and eight planets orbiting around it.",
                "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
                "The Industrial Revolution transformed manufacturing and transportation in the 18th century.",
                
                # Conversation
                "Hello! How can I help you today with your questions?",
                "That's a great question. Let me explain the concept in detail.",
                "I understand your concern. Here are some possible solutions to consider.",
                
                # Creative
                "Once upon a time, in a distant galaxy, there lived a curious explorer.",
                "The sunset painted the sky in brilliant shades of orange and purple.",
                "Innovation drives progress and opens new possibilities for the future.",
                
                # Instructions
                "To solve this task, first identify the key requirements and constraints.",
                "Step by step, we can approach this problem systematically and efficiently.",
                "Let's analyze the situation carefully before making any decisions."
            ]
            
            # Expand to meet sample count
            expanded_prompts = (calibration_prompts * (Config.NUM_CALIB_SAMPLES // len(calibration_prompts) + 1))[:Config.NUM_CALIB_SAMPLES]
            
            # Tokenize
            self.calib_dataset = []
            for prompt in expanded_prompts:
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=Config.MAX_LENGTH
                )
                # Move to device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                self.calib_dataset.append(encoded)
            
            result = f"✅ Calibration dataset prepared: {len(self.calib_dataset)} samples"
            self.log(result)
            return True, result
            
        except Exception as e:
            error_msg = f"❌ Error preparing data: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    # ==================== STAGE 3: BASELINE EVALUATION ====================
    def compute_baseline_metrics(self):
        """Compute baseline perplexity (Section V.A)"""
        try:
            self.log("📊 Computing baseline perplexity...")
            
            if not self.calib_dataset:
                return False, "Please prepare calibration data first"
            
            # Compute perplexity on calibration set
            baseline_ppl = self._compute_perplexity(self.model, self.calib_dataset)
            self.metrics['baseline_perplexity'] = baseline_ppl
            
            # Memory usage
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            self.metrics['baseline_memory_gb'] = memory_gb
            
            result = (
                f"✅ Baseline metrics computed\n"
                f"   Perplexity: {baseline_ppl:.4f}\n"
                f"   Memory: {memory_gb:.2f} GB"
            )
            
            self.log(result)
            return True, result
            
        except Exception as e:
            error_msg = f"❌ Error computing baseline: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    def _compute_perplexity(self, model, dataset, max_batches=None):
        """Compute perplexity with proper loss calculation"""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        batches = dataset if max_batches is None else dataset[:max_batches]
        
        with torch.no_grad():
            for batch in batches:
                # Prepare inputs
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask', None)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                # Accumulate loss
                loss = outputs.loss
                num_tokens = attention_mask.sum().item() if attention_mask is not None else input_ids.numel()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    # ==================== STAGE 4: IMPORTANCE ANALYSIS ====================
    def compute_importance_scores(self):
        """
        Compute weight importance using hybrid metric (Section IV.C, Equation 1)
        I_i = α * ||∂L/∂w_i||_2 + (1-α) * std(A_i)
        """
        try:
            self.log("🔍 Computing importance scores (gradient + activation)...")
            
            if not self.calib_dataset:
                return False, "Please prepare calibration data first"
            
            self.importance_scores = {}
            self.model.train()  # Need gradients
            
            # Identify prunable layers (FFN/MLP layers as per paper)
            prunable_layers = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # Target FFN layers (paper prunes 66% of parameters in FFN)
                    if any(x in name.lower() for x in ['mlp', 'fc', 'dense', 'ffn', 'intermediate']):
                        prunable_layers.append((name, module))
            
            self.log(f"   Found {len(prunable_layers)} prunable layers")
            
            # For each layer, compute importance
            for layer_idx, (name, module) in enumerate(prunable_layers):
                self.log(f"   Analyzing layer {layer_idx+1}/{len(prunable_layers)}: {name}")
                
                # Storage for gradients and activations
                gradients = []
                activations = []
                
                # Hooks to capture data
                def save_activation(mod, inp, out):
                    activations.append(out.detach().cpu())
                
                def save_gradient(mod, grad_in, grad_out):
                    if grad_out[0] is not None:
                        gradients.append(grad_out[0].detach().cpu())
                
                # Register hooks
                fwd_hook = module.register_forward_hook(save_activation)
                bwd_hook = module.register_full_backward_hook(save_gradient)
                
                # Run calibration samples
                num_samples = min(20, len(self.calib_dataset))  # Use subset for efficiency
                
                for batch in self.calib_dataset[:num_samples]:
                    # Forward pass
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch['input_ids']
                    )
                    loss = outputs.loss
                    
                    # Backward pass
                    self.model.zero_grad()
                    loss.backward()
                    
                    # Clear to prevent memory buildup
                    if len(activations) > 100:
                        activations = activations[-50:]
                        gradients = gradients[-50:]
                
                # Remove hooks
                fwd_hook.remove()
                bwd_hook.remove()
                
                # Compute importance score (Equation 1)
                if gradients and activations:
                    # Average gradient magnitude
                    grad_norm = torch.stack([g.norm() for g in gradients]).mean().item()
                    
                    # Average activation std
                    act_std = torch.stack([a.std() for a in activations]).mean().item()
                    
                    # Hybrid importance (Equation 1)
                    importance = Config.ALPHA * grad_norm + (1 - Config.ALPHA) * act_std
                    self.importance_scores[name] = importance
                else:
                    self.log(f"   Warning: No gradients/activations for {name}")
                    self.importance_scores[name] = 0.0
                
                # Cleanup
                del gradients, activations
                gc.collect()
            
            self.model.eval()
            
            result = f"✅ Importance scores computed for {len(self.importance_scores)} layers"
            self.log(result)
            return True, result
            
        except Exception as e:
            self.model.eval()
            error_msg = f"❌ Error computing importance: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    # ==================== STAGE 5: QUBO-GUIDED PRUNING ====================
    def qubo_pruning(self):
        """
        QUBO-guided structured pruning (Section IV.D)
        Objective: H(x) = Σ I_i(1-x_i) + λ(Σx_i - K)²
        Solved via Simulated Annealing
        """
        try:
            self.log("✂️ Performing QUBO-guided pruning...")
            
            if not self.importance_scores:
                return False, "Please compute importance scores first"
            
            total_neurons_pruned = 0
            total_neurons_original = 0
            layers_pruned = 0
            
            for name, module in self.model.named_modules():
                if name not in self.importance_scores:
                    continue
                
                # Get layer weights
                if not hasattr(module, 'weight') or module.weight is None:
                    continue
                    
                weight = module.weight.data
                out_features, in_features = weight.shape
                total_neurons_original += out_features
                
                # Skip tiny layers
                if out_features < 10:
                    self.log(f"   Skipping small layer: {name} ({out_features} neurons)")
                    continue
                
                # Target number of neurons to keep
                K = int(out_features * (1 - Config.PRUNE_RATIO))
                
                self.log(f"   Pruning {name}: {out_features} → {K} neurons")
                
                # Initialize decision variables x_i ∈ {0,1}
                x = torch.ones(out_features, dtype=torch.float32)
                
                # Importance vector (uniform for simplicity, could weight by importance)
                importance = torch.full((out_features,), self.importance_scores[name])
                
                # Simulated Annealing to solve QUBO (Algorithm in Section IV.D)
                T = Config.T0
                best_x = x.clone()
                best_energy = self._qubo_energy(x, importance, K, Config.LAMBDA)
                
                iteration = 0
                while T > Config.T_MIN and iteration < Config.MAX_SA_ITERATIONS:
                    # Inner loop: propose random flips
                    for _ in range(Config.SA_ITERATIONS_PER_TEMP):
                        # Randomly flip one bit
                        i = random.randint(0, out_features - 1)
                        x_new = x.clone()
                        x_new[i] = 1 - x_new[i]
                        
                        # Compute energy change
                        energy_new = self._qubo_energy(x_new, importance, K, Config.LAMBDA)
                        delta_E = energy_new - best_energy
                        
                        # Accept with Metropolis criterion
                        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                            x = x_new.clone()
                            if energy_new < best_energy:
                                best_x = x.clone()
                                best_energy = energy_new
                    
                    # Cool down
                    T *= Config.ALPHA_COOLING
                    iteration += Config.SA_ITERATIONS_PER_TEMP
                
                # Apply pruning mask
                mask = best_x > 0.5
                neurons_kept = mask.sum().item()
                neurons_pruned = out_features - neurons_kept
                
                if neurons_pruned > 0:
                    # Structured pruning: remove entire neurons (rows)
                    module.weight.data = weight[mask]
                    if module.bias is not None:
                        module.bias.data = module.bias.data[mask]
                    
                    self.pruning_masks[name] = mask
                    total_neurons_pruned += neurons_pruned
                    layers_pruned += 1
                    
                    self.log(f"      Pruned {neurons_pruned} neurons (kept {neurons_kept}/{out_features})")
            
            # Compute metrics after pruning
            pruned_params = sum(p.numel() for p in self.model.parameters())
            self.metrics['pruned_params'] = pruned_params
            self.metrics['neurons_pruned'] = total_neurons_pruned
            self.metrics['layers_pruned'] = layers_pruned
            self.metrics['param_reduction_pct'] = (1 - pruned_params / self.metrics['original_params']) * 100
            
            # Evaluate quality degradation
            post_prune_ppl = self._compute_perplexity(self.model, self.calib_dataset[:10])
            self.metrics['post_prune_perplexity'] = post_prune_ppl
            
            result = (
                f"✅ QUBO pruning complete\n"
                f"   Layers pruned: {layers_pruned}\n"
                f"   Neurons removed: {total_neurons_pruned:,}\n"
                f"   Parameter reduction: {self.metrics['param_reduction_pct']:.2f}%\n"
                f"   Post-prune perplexity: {post_prune_ppl:.4f}"
            )
            
            self.log(result)
            return True, result
            
        except Exception as e:
            error_msg = f"❌ Error in QUBO pruning: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    def _qubo_energy(self, x, I, K, lam):
        """
        Compute QUBO energy function (Equation 2)
        H(x) = Σ I_i(1-x_i) + λ(Σx_i - K)²
        """
        importance_term = torch.sum(I * (1 - x))
        sparsity_penalty = lam * (torch.sum(x) - K) ** 2
        return importance_term + sparsity_penalty
    
    # ==================== STAGE 6: FINE-TUNING ====================
    def fine_tune(self):
        """
        Fine-tune to recover performance (Section IV.E)
        Uses instruction-following data with AdamW optimizer
        """
        try:
            self.log("🔧 Fine-tuning for performance recovery...")
            
            # Instruction-following dataset (Alpaca-style)
            instruction_data = [
                "Instruction: Explain neural networks. Response: Neural networks are computational models inspired by biological neurons that learn patterns from data through training.",
                "Instruction: What is machine learning? Response: Machine learning enables computers to learn and improve from experience without explicit programming.",
                "Instruction: Define artificial intelligence. Response: Artificial intelligence is the simulation of human intelligence by machines to perform tasks requiring cognition.",
                "Instruction: How does compression work? Response: Compression reduces model size through techniques like pruning redundant parameters and quantizing numerical precision.",
                "Instruction: Explain transformers. Response: Transformers are neural architectures using self-attention mechanisms to process sequential data efficiently.",
                "Instruction: What is pruning? Response: Pruning removes unnecessary neural network parameters to reduce model size while maintaining performance.",
                "Instruction: Describe quantization. Response: Quantization converts high-precision numbers to lower precision formats, reducing memory and computation requirements.",
                "Instruction: How do LLMs work? Response: Large language models process text using transformer architectures with billions of parameters trained on vast text corpora.",
                "Instruction: What is edge AI? Response: Edge AI performs artificial intelligence computations on local devices rather than cloud servers for privacy and speed.",
                "Instruction: Explain optimization. Response: Optimization finds the best solution from available alternatives using mathematical algorithms and objective functions.",
            ] * 20  # Expand dataset
            
            # Create dataset
            class InstructionDataset(Dataset):
                def __init__(self, texts, tokenizer):
                    self.encodings = []
                    for text in texts:
                        enc = tokenizer(
                            text,
                            truncation=True,
                            max_length=128,
                            padding='max_length',
                            return_tensors='pt'
                        )
                        self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
                
                def __len__(self):
                    return len(self.encodings)
                
                def __getitem__(self, idx):
                    return self.encodings[idx]
            
            dataset = InstructionDataset(instruction_data, self.tokenizer)
            dataloader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True
            )
            
            # Setup optimizer (Section IV.E parameters)
            optimizer = AdamW(
                self.model.parameters(),
                lr=Config.LEARNING_RATE,
                weight_decay=0.01
            )
            
            # Learning rate schedule with warmup
            total_steps = min(Config.FINE_TUNE_STEPS, len(dataloader))
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=Config.WARMUP_STEPS,
                num_training_steps=total_steps
            )
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            
            for step, batch in enumerate(dataloader):
                if step >= Config.FINE_TUNE_STEPS:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['input_ids']
                )
                loss = outputs.loss / Config.GRADIENT_ACCUM_STEPS
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % Config.GRADIENT_ACCUM_STEPS == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * Config.GRADIENT_ACCUM_STEPS
                
                # Logging
                if (step + 1) % 20 == 0:
                    avg_loss = total_loss / (step + 1)
                    self.log(f"      Step {step+1}/{Config.FINE_TUNE_STEPS}, Loss: {avg_loss:.4f}")
            
            self.model.eval()
            
            # Evaluate post-fine-tuning
            avg_loss = total_loss / min(Config.FINE_TUNE_STEPS, len(dataloader))
            post_ft_ppl = self._compute_perplexity(self.model, self.calib_dataset[:10])
            
            self.metrics['fine_tune_loss'] = avg_loss
            self.metrics['post_finetune_perplexity'] = post_ft_ppl
            
            result = (
                f"✅ Fine-tuning complete\n"
                f"   Training loss: {avg_loss:.4f}\n"
                f"   Post-FT perplexity: {post_ft_ppl:.4f}"
            )
            
            self.log(result)
            return True, result
            
        except Exception as e:
            self.model.eval()
            error_msg = f"❌ Error in fine-tuning: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    # ==================== STAGE 7: QUANTIZATION ====================
    def quantize_model(self):
        """
        INT8 dynamic quantization (Section IV.F, Table III)
        Converts FP32 weights to INT8 for 4x compression
        """
        try:
            self.log("⚡ Applying INT8 quantization...")
            
            # Move to CPU for quantization
            self.model = self.model.cpu()
            
            # Apply dynamic quantization (weights to INT8, activations stay FP32)
            self.quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear},  # Quantize all Linear layers
                dtype=Config.QUANT_DTYPE
            )
            
            # Calculate final metrics
            quantized_size = self._calculate_model_size(self.quantized_model)
            self.metrics['quantized_size_gb'] = quantized_size
            self.metrics['compression_ratio'] = self.metrics['original_size_gb'] / quantized_size
            self.metrics['size_reduction_pct'] = (1 - quantized_size / self.metrics['original_size_gb']) * 100
            
            # Final quality evaluation
            final_ppl = self._compute_perplexity(self.quantized_model, self.calib_dataset[:10])
            self.metrics['final_perplexity'] = final_ppl
            self.metrics['perplexity_increase_pct'] = ((final_ppl - self.metrics['baseline_perplexity']) / self.metrics['baseline_perplexity']) * 100
            self.metrics['quality_retention'] = (self.metrics['baseline_perplexity'] / final_ppl) * 100
            
            # Save compressed model
            output_path = Path(Config.OUTPUT_DIR)
            self.log(f"   Saving compressed model to {output_path}")
            
            # Save quantized model (PyTorch format)
            torch.save(self.quantized_model.state_dict(), output_path / "quantized_model.pth")
            self.tokenizer.save_pretrained(output_path)
            
            # Save metrics
            with open(output_path / Config.METRICS_FILE, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            # Generate summary report
            report = self._generate_compression_report()
            with open(output_path / "compression_report.txt", 'w') as f:
                f.write(report)
            
            result = (
                f"✅ Quantization complete\n"
                f"   Original size: {self.metrics['original_size_gb']:.3f} GB\n"
                f"   Compressed size: {quantized_size:.3f} GB\n"
                f"   Compression ratio: {self.metrics['compression_ratio']:.2f}x\n"
                f"   Size reduction: {self.metrics['size_reduction_pct']:.1f}%\n"
                f"   Final perplexity: {final_ppl:.4f}\n"
                f"   Quality retention: {self.metrics['quality_retention']:.1f}%\n"
                f"   Model saved to: {output_path}"
            )
            
            self.log(result)
            return True, result
            
        except Exception as e:
            error_msg = f"❌ Error in quantization: {str(e)}"
            self.log(error_msg)
            return False, error_msg
    
    def _generate_compression_report(self):
        """Generate detailed compression report matching paper format"""
        report = f"""
{'='*70}
HYBRID LLM COMPRESSION PIPELINE - FINAL REPORT
{'='*70}

Model: {self.metrics.get('model_name', 'N/A')}
Device: {self.metrics.get('device', 'N/A')}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

COMPRESSION BREAKDOWN (Table IV Style)
{'-'*70}
Stage                    Size (GB)    Reduction    Perplexity
{'-'*70}
Original (FP32)          {self.metrics.get('original_size_gb', 0):.3f}        0.0%         {self.metrics.get('baseline_perplexity', 0):.4f}
After Pruning            {self.metrics.get('original_size_gb', 0):.3f}        {self.metrics.get('param_reduction_pct', 0):.1f}%        {self.metrics.get('post_prune_perplexity', 0):.4f}
+ Fine-Tuning            {self.metrics.get('original_size_gb', 0):.3f}        {self.metrics.get('param_reduction_pct', 0):.1f}%        {self.metrics.get('post_finetune_perplexity', 0):.4f}
+ Quantization (INT8)    {self.metrics.get('quantized_size_gb', 0):.3f}        {self.metrics.get('size_reduction_pct', 0):.1f}%        {self.metrics.get('final_perplexity', 0):.4f}
{'-'*70}

DETAILED METRICS
{'-'*70}
Parameters:
  Original:              {self.metrics.get('original_params', 0):,}
  After Pruning:         {self.metrics.get('pruned_params', 0):,}
  Neurons Pruned:        {self.metrics.get('neurons_pruned', 0):,}
  Layers Pruned:         {self.metrics.get('layers_pruned', 0)}

Compression:
  Compression Ratio:     {self.metrics.get('compression_ratio', 0):.2f}x
  Size Reduction:        {self.metrics.get('size_reduction_pct', 0):.1f}%

Quality:
  Baseline Perplexity:   {self.metrics.get('baseline_perplexity', 0):.4f}
  Final Perplexity:      {self.metrics.get('final_perplexity', 0):.4f}
  Perplexity Increase:   {self.metrics.get('perplexity_increase_pct', 0):.1f}%
  Quality Retention:     {self.metrics.get('quality_retention', 0):.1f}%

CONFIGURATION
{'-'*70}
Pruning Ratio:           {Config.PRUNE_RATIO:.0%}
QUBO Lambda:             {Config.LAMBDA}
Alpha (Importance):      {Config.ALPHA}
Fine-tune Steps:         {Config.FINE_TUNE_STEPS}
Learning Rate:           {Config.LEARNING_RATE}
Quantization:            INT8 Dynamic

{'='*70}
Report generated by Hybrid Compression Pipeline
Based on research paper methodology
{'='*70}
"""
        return report
    
    # ==================== INFERENCE ====================
    def generate_text(self, prompt, max_new_tokens=50):
        """Generate text using compressed model"""
        try:
            if self.quantized_model is None:
                return "❌ Please complete the compression pipeline first"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=Config.MAX_LENGTH
            )
            
            # Generate
            self.quantized_model.eval()
            with torch.no_grad():
                outputs = self.quantized_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return only the new generated part
            response = generated_text[len(prompt):].strip()
            return response if response else generated_text
            
        except Exception as e:
            return f"❌ Generation error: {str(e)}"


# ==================== PROFESSIONAL GUI ====================
class CompressionGUI:
    """Professional GUI for the compression pipeline"""
    
    def __init__(self, root):
        self.root = root
        self.pipeline = HybridCompressionPipeline(callback=self.log_message)
        self.is_running = False
        self.setup_ui()
        
    def setup_ui(self):
        """Create professional UI layout"""
        self.root.title("LLM Compression Pipeline - Research Implementation")
        self.root.geometry("1000x750")
        self.root.configure(bg='#2c3e50')
        
        # Main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ==================== HEADER ====================
        header_frame = tk.Frame(main_container, bg='#34495e', relief='raised', bd=2)
        header_frame.pack(fill='x', pady=(0, 10))
        
        title_label = tk.Label(
            header_frame,
            text="🔬 Hybrid LLM Compression Pipeline",
            font=('Helvetica', 18, 'bold'),
            bg='#34495e',
            fg='#ecf0f1',
            pady=10
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="QUBO-Guided Pruning + Fine-Tuning + INT8 Quantization",
            font=('Helvetica', 10),
            bg='#34495e',
            fg='#bdc3c7'
        )
        subtitle_label.pack(pady=(0, 10))
        
        # ==================== CONTROL PANEL ====================
        control_frame = ttk.LabelFrame(main_container, text="Pipeline Controls", padding=15)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill='x', pady=(0, 10))
        
        self.status_label = tk.Label(
            control_frame,
            text="Ready to start compression pipeline",
            font=('Helvetica', 10),
            fg='#27ae60',
            bg='#ecf0f1'
        )
        self.status_label.pack(pady=(0, 10))
        
        # Button grid
        button_frame = tk.Frame(control_frame, bg='#ecf0f1')
        button_frame.pack(fill='x')
        
        # Define pipeline stages
        self.stages = [
            ("1. Load Model", self.stage_load_model, 14.3),
            ("2. Prepare Data", self.stage_prepare_data, 28.6),
            ("3. Baseline Metrics", self.stage_baseline, 42.9),
            ("4. Importance Analysis", self.stage_importance, 57.1),
            ("5. QUBO Pruning", self.stage_pruning, 71.4),
            ("6. Fine-Tune", self.stage_finetune, 85.7),
            ("7. Quantize", self.stage_quantize, 100.0),
        ]
        
        # Create stage buttons
        for i, (label, command, progress) in enumerate(self.stages):
            btn = tk.Button(
                button_frame,
                text=label,
                command=command,
                font=('Helvetica', 10, 'bold'),
                bg='#3498db',
                fg='white',
                activebackground='#2980b9',
                relief='raised',
                bd=2,
                width=18,
                height=2,
                cursor='hand2'
            )
            btn.grid(row=i//4, column=i%4, padx=5, pady=5, sticky='ew')
        
        # Configure grid weights
        for i in range(4):
            button_frame.columnconfigure(i, weight=1)
        
        # Full pipeline button
        full_pipeline_btn = tk.Button(
            control_frame,
            text="▶ RUN FULL PIPELINE",
            command=self.run_full_pipeline,
            font=('Helvetica', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            activebackground='#229954',
            relief='raised',
            bd=3,
            height=2,
            cursor='hand2'
        )
        full_pipeline_btn.pack(fill='x', pady=(10, 0))
        
        # ==================== METRICS DISPLAY ====================
        metrics_frame = ttk.LabelFrame(main_container, text="Compression Metrics", padding=10)
        metrics_frame.pack(fill='x', pady=(0, 10))
        
        self.metrics_text = tk.Text(
            metrics_frame,
            height=6,
            font=('Courier', 9),
            bg='#1c1c1c',
            fg='#00ff00',
            relief='sunken',
            bd=2
        )
        self.metrics_text.pack(fill='both', expand=True)
        self.update_metrics_display()
        
        # ==================== LOG CONSOLE ====================
        log_frame = ttk.LabelFrame(main_container, text="Execution Log", padding=10)
        log_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=('Consolas', 9),
            bg='#1e1e1e',
            fg='#d4d4d4',
            relief='sunken',
            bd=2
        )
        self.log_text.pack(fill='both', expand=True)
        
        # ==================== TEST GENERATION ====================
        test_frame = ttk.LabelFrame(main_container, text="Test Compressed Model", padding=10)
        test_frame.pack(fill='x')
        
        prompt_frame = tk.Frame(test_frame)
        prompt_frame.pack(fill='x', pady=(0, 5))
        
        tk.Label(prompt_frame, text="Prompt:", font=('Helvetica', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        self.prompt_entry = ttk.Entry(prompt_frame, font=('Helvetica', 10))
        self.prompt_entry.insert(0, "Explain neural network compression:")
        self.prompt_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        generate_btn = tk.Button(
            prompt_frame,
            text="Generate",
            command=self.test_generation,
            font=('Helvetica', 10, 'bold'),
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            cursor='hand2',
            width=12
        )
        generate_btn.pack(side='left')
        
        # Initial log message
        self.log_message("="*80)
        self.log_message("🚀 Hybrid LLM Compression Pipeline Initialized")
        self.log_message("Based on: 'A Hybrid Approach to Compressing Large Language Models'")
        self.log_message(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        self.log_message("="*80)
    
    # ==================== LOGGING ====================
    def log_message(self, message):
        """Add timestamped message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, message, color='#27ae60'):
        """Update status label"""
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.root.update_idletasks()
    
    def update_metrics_display(self):
        """Update metrics display panel"""
        self.metrics_text.delete('1.0', tk.END)
        
        metrics = self.pipeline.metrics
        if not metrics:
            self.metrics_text.insert('1.0', "No metrics available yet. Run the pipeline to see results.")
            return
        
        # Format metrics display
        display = []
        display.append("┌─ COMPRESSION SUMMARY ─────────────────────────────────────────────┐")
        
        if 'original_size_gb' in metrics:
            display.append(f"│ Original Size:      {metrics['original_size_gb']:.3f} GB")
        if 'quantized_size_gb' in metrics:
            display.append(f"│ Compressed Size:    {metrics['quantized_size_gb']:.3f} GB")
        if 'compression_ratio' in metrics:
            display.append(f"│ Compression Ratio:  {metrics['compression_ratio']:.2f}x")
        if 'size_reduction_pct' in metrics:
            display.append(f"│ Size Reduction:     {metrics['size_reduction_pct']:.1f}%")
        
        display.append("├───────────────────────────────────────────────────────────────────┤")
        
        if 'baseline_perplexity' in metrics:
            display.append(f"│ Baseline PPL:       {metrics['baseline_perplexity']:.4f}")
        if 'final_perplexity' in metrics:
            display.append(f"│ Final PPL:          {metrics['final_perplexity']:.4f}")
        if 'quality_retention' in metrics:
            display.append(f"│ Quality Retention:  {metrics['quality_retention']:.1f}%")
        
        if 'neurons_pruned' in metrics:
            display.append("├───────────────────────────────────────────────────────────────────┤")
            display.append(f"│ Neurons Pruned:     {metrics['neurons_pruned']:,}")
            display.append(f"│ Layers Pruned:      {metrics.get('layers_pruned', 0)}")
        
        display.append("└───────────────────────────────────────────────────────────────────┘")
        
        self.metrics_text.insert('1.0', '\n'.join(display))
    
    # ==================== STAGE EXECUTION ====================
    def run_in_thread(self, func, stage_name, progress_value):
        """Execute stage in separate thread"""
        if self.is_running:
            messagebox.showwarning("Warning", "Pipeline is already running!")
            return
        
        def wrapper():
            self.is_running = True
            self.update_status(f"Running: {stage_name}", '#e67e22')
            self.update_progress(progress_value)
            
            try:
                success, message = func()
                if success:
                    self.update_status(f"✓ Completed: {stage_name}", '#27ae60')
                    self.update_metrics_display()
                else:
                    self.update_status(f"✗ Failed: {stage_name}", '#e74c3c')
            except Exception as e:
                self.log_message(f"❌ Unexpected error: {str(e)}")
                self.update_status(f"✗ Error: {stage_name}", '#e74c3c')
            finally:
                self.is_running = False
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
    
    def stage_load_model(self):
        self.run_in_thread(self.pipeline.load_model, "Load Model", 14.3)
    
    def stage_prepare_data(self):
        self.run_in_thread(self.pipeline.prepare_calibration_data, "Prepare Data", 28.6)
    
    def stage_baseline(self):
        self.run_in_thread(self.pipeline.compute_baseline_metrics, "Baseline Metrics", 42.9)
    
    def stage_importance(self):
        self.run_in_thread(self.pipeline.compute_importance_scores, "Importance Analysis", 57.1)
    
    def stage_pruning(self):
        self.run_in_thread(self.pipeline.qubo_pruning, "QUBO Pruning", 71.4)
    
    def stage_finetune(self):
        self.run_in_thread(self.pipeline.fine_tune, "Fine-Tuning", 85.7)
    
    def stage_quantize(self):
        self.run_in_thread(self.pipeline.quantize_model, "Quantization", 100.0)
    
    # ==================== FULL PIPELINE ====================
    def run_full_pipeline(self):
        """Execute complete compression pipeline"""
        if self.is_running:
            messagebox.showwarning("Warning", "Pipeline is already running!")
            return
        
        # Confirm
        response = messagebox.askyesno(
            "Run Full Pipeline",
            "This will run all 7 stages of the compression pipeline.\n"
            "This may take 15-30 minutes depending on your hardware.\n\n"
            "Continue?"
        )
        
        if not response:
            return
        
        def run_all_stages():
            self.is_running = True
            start_time = time.time()
            
            self.log_message("\n" + "="*80)
            self.log_message("🚀 STARTING FULL COMPRESSION PIPELINE")
            self.log_message("="*80 + "\n")
            
            # Execute all stages
            for i, (label, command, progress) in enumerate(self.stages):
                stage_name = label.split('. ')[1]
                self.log_message(f"\n{'─'*80}")
                self.log_message(f"▶ Stage {i+1}/7: {stage_name}")
                self.log_message(f"{'─'*80}")
                
                self.update_status(f"Stage {i+1}/7: {stage_name}", '#e67e22')
                self.update_progress(progress)
                
                # Get the actual pipeline method
                method_map = {
                    "Load Model": self.pipeline.load_model,
                    "Prepare Data": self.pipeline.prepare_calibration_data,
                    "Baseline Metrics": self.pipeline.compute_baseline_metrics,
                    "Importance Analysis": self.pipeline.compute_importance_scores,
                    "QUBO Pruning": self.pipeline.qubo_pruning,
                    "Fine-Tune": self.pipeline.fine_tune,
                    "Quantize": self.pipeline.quantize_model,
                }
                
                method = method_map[stage_name]
                
                try:
                    success, message = method()
                    if not success:
                        self.log_message(f"\n❌ Pipeline stopped at stage: {stage_name}")
                        self.update_status(f"Failed at: {stage_name}", '#e74c3c')
                        break
                    
                    self.update_metrics_display()
                    time.sleep(1)  # Brief pause between stages
                    
                except Exception as e:
                    self.log_message(f"\n❌ Error in stage {stage_name}: {str(e)}")
                    self.update_status(f"Error at: {stage_name}", '#e74c3c')
                    break
            else:
                # All stages completed successfully
                elapsed_time = time.time() - start_time
                self.log_message("\n" + "="*80)
                self.log_message("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
                self.log_message(f"Total time: {elapsed_time/60:.1f} minutes")
                self.log_message("="*80 + "\n")
                
                self.update_status("✓ All stages completed successfully!", '#27ae60')
                self.update_progress(100)
                
                # Show completion dialog
                messagebox.showinfo(
                    "Success",
                    f"Compression pipeline completed!\n\n"
                    f"Time: {elapsed_time/60:.1f} minutes\n"
                    f"Compressed model saved to: {Config.OUTPUT_DIR}\n\n"
                    f"Check the metrics panel for results."
                )
            
            self.is_running = False
        
        # Run in thread
        thread = threading.Thread(target=run_all_stages, daemon=True)
        thread.start()
    
    # ==================== TEST GENERATION ====================
    def test_generation(self):
        """Test the compressed model with user prompt"""
        if self.pipeline.quantized_model is None:
            messagebox.showwarning(
                "Model Not Ready",
                "Please complete the compression pipeline first!"
            )
            return
        
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a prompt!")
            return
        
        self.log_message(f"\n{'─'*80}")
        self.log_message(f"🤖 Generating response for: '{prompt}'")
        self.log_message(f"{'─'*80}")
        
        def generate():
            self.update_status("Generating...", '#9b59b6')
            response = self.pipeline.generate_text(prompt, max_new_tokens=50)
            self.log_message(f"Response: {response}\n")
            self.update_status("Generation complete", '#27ae60')
        
        thread = threading.Thread(target=generate, daemon=True)
        thread.start()


# ==================== MAIN ====================
def main():
    """Launch the application"""
    root = tk.Tk()
    app = CompressionGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Handle window close
    def on_closing():
        if app.is_running:
            if messagebox.askokcancel("Quit", "Pipeline is running. Are you sure you want to quit?"):
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    print("="*80)
    print("Hybrid LLM Compression Pipeline")
    print("Research Implementation")
    print("="*80)
    print("\nStarting GUI...")
    main()