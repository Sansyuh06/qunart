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

# ==================== HARDWARE DETECTION ====================
def _detect_hardware():
    """Auto-detect hardware capabilities and return a summary dict."""
    info = {
        'has_cuda': torch.cuda.is_available(),
        'gpu_name': None,
        'gpu_count': 0,
        'vram_gb': 0.0,
        'cpu_cores': psutil.cpu_count(logical=False) or 1,
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'use_fp16': False,
    }
    if info['has_cuda']:
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Enable FP16 for GPUs with compute capability >= 7.0 (Volta+)
        cc = torch.cuda.get_device_capability(0)
        info['use_fp16'] = (cc[0] >= 7)
    return info


HW_INFO = _detect_hardware()


# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters — auto-tuned to the detected hardware."""

    # Model Selection (Table II)
    MODEL_NAME = "microsoft/DialoGPT-medium"  # 354M params

    # Compression Parameters (Section IV)
    PRUNE_RATIO = 0.20              # 20% neuron pruning (structured)
    ALPHA = 0.6                      # Gradient vs activation weight (Eq. 1)
    LAMBDA = 1.5                     # QUBO penalty parameter (Eq. 2)

    # ── Auto-tuned by hardware ──
    if HW_INFO['has_cuda'] and HW_INFO['vram_gb'] >= 6:
        # GPU with ≥ 6 GB VRAM → bigger batches, more calibration
        NUM_CALIB_SAMPLES = 200
        FINE_TUNE_STEPS   = 150
        BATCH_SIZE        = 4
        GRADIENT_ACCUM_STEPS = 2   # Effective batch: 8
    elif HW_INFO['has_cuda']:
        # Smaller GPU → moderate settings
        NUM_CALIB_SAMPLES = 100
        FINE_TUNE_STEPS   = 100
        BATCH_SIZE        = 2
        GRADIENT_ACCUM_STEPS = 2   # Effective batch: 4
    else:
        # CPU-only → conservative
        NUM_CALIB_SAMPLES = 50
        FINE_TUNE_STEPS   = 60
        BATCH_SIZE        = 1
        GRADIENT_ACCUM_STEPS = 4   # Effective batch: 4

    LEARNING_RATE = 1e-5
    WARMUP_STEPS  = 10

    # QUBO / Simulated Annealing (Section IV.D)
    T0 = 10.0
    T_MIN = 0.01
    ALPHA_COOLING = 0.95
    SA_ITERATIONS_PER_TEMP = 100
    MAX_SA_ITERATIONS = 400

    # Evaluation
    MAX_LENGTH = 256
    QUALITY_THRESHOLD = 0.95

    # Quantization (Section IV.F, Table III)
    QUANT_DTYPE = torch.qint8       # INT8 quantization

    # Mixed precision — enabled automatically on capable GPUs
    USE_FP16 = HW_INFO['use_fp16']

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
        self.hw = HW_INFO  # hardware info for downstream logic
        
        # Create output directory
        Path(Config.OUTPUT_DIR).mkdir(exist_ok=True)
        
    def log(self, message):
        """Send log message to GUI"""
        if self.callback:
            self.callback(message)
        print(message)
    
    def _gpu_cleanup(self):
        """Free unused GPU memory between heavy stages."""
        if self.hw['has_cuda']:
            torch.cuda.empty_cache()
            gc.collect()
    
    # ==================== STAGE 1: MODEL LOADING ====================
    def load_model(self):
        """Load pre-trained model and tokenizer (Section IV.B)"""
        try:
            self._gpu_cleanup()
            self.log("📦 Loading model and tokenizer...")
            
            # Log detected hardware
            if self.hw['has_cuda']:
                self.log(f"   🖥️  GPU detected: {self.hw['gpu_name']}")
                self.log(f"   💾 VRAM: {self.hw['vram_gb']:.1f} GB  |  FP16: {'ON' if Config.USE_FP16 else 'OFF'}")
            else:
                self.log(f"   🖥️  Running on CPU ({self.hw['cpu_cores']} cores, {self.hw['ram_gb']:.1f} GB RAM)")
            self.log(f"   ⚙️  Auto-tuned: batch={Config.BATCH_SIZE}, calib={Config.NUM_CALIB_SAMPLES}, steps={Config.FINE_TUNE_STEPS}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                Config.MODEL_NAME,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model — attempt GPU, fall back to CPU on OOM
            load_dtype = torch.float32  # FP32 needed for pruning accuracy
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    Config.MODEL_NAME,
                    torch_dtype=load_dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                self.model.to(self.device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom:
                if self.hw['has_cuda']:
                    self.log("   ⚠️  GPU out of memory — falling back to CPU")
                    self._gpu_cleanup()
                    self.device = torch.device('cpu')
                    self.model = AutoModelForCausalLM.from_pretrained(
                        Config.MODEL_NAME,
                        torch_dtype=load_dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                else:
                    raise oom
            
            self.model.eval()
            
            # Calculate baseline metrics
            param_count = sum(p.numel() for p in self.model.parameters())
            self.metrics['original_params'] = param_count
            self.metrics['original_size_gb'] = self._calculate_model_size(self.model)
            self.metrics['device'] = str(self.device)
            self.metrics['model_name'] = Config.MODEL_NAME
            
            # Log GPU memory after loading
            if self.hw['has_cuda'] and self.device.type == 'cuda':
                alloc = torch.cuda.memory_allocated() / (1024**3)
                self.log(f"   📊 GPU memory used: {alloc:.2f} / {self.hw['vram_gb']:.1f} GB")
            
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
            
            self._gpu_cleanup()
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
        
        # Determine the device of the model dynamically
        device = next(model.parameters()).device
        
        batches = dataset if max_batches is None else dataset[:max_batches]
        
        with torch.no_grad():
            for batch in batches:
                # Prepare inputs and move to model's device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device) if batch.get('attention_mask') is not None else None
                
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
            
            # Build connection mapping: map name to its successor module
            all_modules = list(self.model.named_modules())
            self.successor_map = {}
            for i, (name, module) in enumerate(all_modules):
                if not (isinstance(module, nn.Linear) or type(module).__name__ == 'Conv1D'):
                    continue
                if any(x in name.lower() for x in ['mlp', 'fc', 'dense', 'ffn', 'intermediate']):
                    for j in range(i + 1, len(all_modules)):
                        next_name, next_module = all_modules[j]
                        if isinstance(next_module, nn.Linear) or type(next_module).__name__ == 'Conv1D':
                            parent_path = ".".join(name.split(".")[:-1])
                            next_parent_path = ".".join(next_name.split(".")[:-1])
                            if parent_path == next_parent_path:
                                self.successor_map[name] = (next_name, next_module)
                            break
            
            # Identify prunable layers (FFN/MLP layers as per paper)
            prunable_layers = []
            for name, module in self.model.named_modules():
                if name in self.successor_map:
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
                
                try:
                    # Run calibration samples
                    num_samples = min(20, len(self.calib_dataset))  # Use subset for efficiency
                    use_amp = Config.USE_FP16 and self.device.type == 'cuda'
                    
                    for batch in self.calib_dataset[:num_samples]:
                        device = next(self.model.parameters()).device
                        input_ids = batch['input_ids'].to(device)
                        attn_mask = batch.get('attention_mask')
                        attn_mask = attn_mask.to(device) if attn_mask is not None else None
                        
                        # Forward pass (with optional AMP for GPU users)
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(
                                    input_ids=input_ids,
                                    attention_mask=attn_mask,
                                    labels=input_ids
                                )
                                loss = outputs.loss.float()  # upcast for backward
                        else:
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attn_mask,
                                labels=input_ids
                            )
                            loss = outputs.loss
                        
                        # Backward pass
                        self.model.zero_grad()
                        loss.backward()
                        
                        # Clear to prevent memory buildup
                        if len(activations) > 100:
                            activations = activations[-50:]
                            gradients = gradients[-50:]
                finally:
                    # Remove hooks defensively
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
                self._gpu_cleanup()
            
            self.model.eval()
            self._gpu_cleanup()
            
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
                is_conv1d = type(module).__name__ == 'Conv1D'
                if is_conv1d:
                    in_features, out_features = weight.shape
                else:
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
                current_energy = best_energy
                
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
                        delta_E = energy_new - current_energy
                        
                        # Accept with Metropolis criterion
                        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                            x = x_new.clone()
                            current_energy = energy_new
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
                    # Structured pruning: remove entire neurons (rows for Linear, cols for Conv1D)
                    if is_conv1d:
                        module.weight.data = weight[:, mask]
                    else:
                        module.weight.data = weight[mask]
                        
                    if module.bias is not None:
                        module.bias.data = module.bias.data[mask]
                    
                    # Adjust successor layer's input dimensions using the same mask
                    if hasattr(self, 'successor_map') and name in self.successor_map:
                        successor_name, successor_module = self.successor_map[name]
                        is_succ_conv1d = type(successor_module).__name__ == 'Conv1D'
                        if is_succ_conv1d:
                            successor_module.weight.data = successor_module.weight.data[mask, :]
                        else:
                            successor_module.weight.data = successor_module.weight.data[:, mask]
                    
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
        Supports mixed-precision training on capable GPUs
        """
        try:
            self._gpu_cleanup()
            self.log("🔧 Fine-tuning for performance recovery...")
            
            use_amp = Config.USE_FP16 and self.device.type == 'cuda'
            if use_amp:
                self.log("   ⚡ Mixed-precision (FP16) training enabled")
            
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
            num_training_steps = Config.FINE_TUNE_STEPS // Config.GRADIENT_ACCUM_STEPS
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=Config.WARMUP_STEPS,
                num_training_steps=num_training_steps
            )
            
            # AMP scaler for mixed-precision training
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            
            # Training loop
            self.model.train()
            total_loss = 0.0
            
            for step, batch in enumerate(dataloader):
                if step >= Config.FINE_TUNE_STEPS:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass — with optional AMP
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch.get('attention_mask'),
                        labels=batch['input_ids']
                    )
                    loss = outputs.loss / Config.GRADIENT_ACCUM_STEPS
                
                # Backward pass — scaled when using AMP
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % Config.GRADIENT_ACCUM_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * Config.GRADIENT_ACCUM_STEPS
                
                # Logging
                if (step + 1) % 20 == 0:
                    avg_loss = total_loss / (step + 1)
                    self.log(f"      Step {step+1}/{Config.FINE_TUNE_STEPS}, Loss: {avg_loss:.4f}")
            
            self.model.eval()
            self._gpu_cleanup()
            
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
            self._gpu_cleanup()
            self.log("⚡ Applying INT8 quantization...")
            
            # Move to CPU for quantization (PyTorch quantization requires CPU)
            if self.device.type == 'cuda':
                self.log("   Moving model to CPU for quantization...")
            self.model = self.model.cpu()
            self._gpu_cleanup()  # free GPU VRAM now that model is on CPU
            
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


# ==================== EEL BACKEND ====================
import eel

# Initialize Eel and point it to the 'web' directory
eel.init('web')

# Instantiate the pipeline globally so Eel can interact with it
# We wrap the log method so it forwards logs to JS
def eel_log_callback(msg):
    eel.ui_log(msg)()

pipeline = HybridCompressionPipeline(callback=eel_log_callback)

@eel.expose
def get_sys_stats():
    """Return real-time CPU, RAM, and GPU stats to JS."""
    try:
        proc = psutil.Process()
        mem = proc.memory_info()
        cpu = psutil.cpu_percent(interval=None)
        ram = mem.rss / (1024**3)
        ram_total = psutil.virtual_memory().total / (1024**3)
        ram_pct = (ram / ram_total) * 100
        
        gpu_pct = 0.0
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_pct = (alloc / total) * 100
            
        return {'cpu': cpu, 'ram': ram_pct, 'gpu': gpu_pct}
    except Exception:
        return {'cpu': 0, 'ram': 0, 'gpu': 0}

@eel.expose
def get_initial_hardware():
    """Send initial hardware configuration to JS."""
    eel.ui_set_hardware(pipeline.hw['gpu_name'] if pipeline.hw['gpu_name'] else 'CPU', pipeline.hw['has_cuda'])()

@eel.expose
def start_pipeline():
    """Run the entire compression pipeline sequentially and update JS."""
    stages = [
        (pipeline.load_model, "Load Model"),
        (pipeline.prepare_calibration_data, "Prepare Data"),
        (pipeline.compute_baseline_metrics, "Baseline Metrics"),
        (pipeline.compute_importance_scores, "Importance Analysis"),
        (pipeline.qubo_pruning, "QUBO Pruning"),
        (pipeline.fine_tune, "Fine-Tuning"),
        (pipeline.quantize_model, "Quantization")
    ]
    
    for i, (method, name) in enumerate(stages):
        eel.ui_set_stage(i + 1)()
        pipeline.log(f"\n▶ Starting Stage {i + 1}: {name}")
        
        try:
            success, msg = method()
            if not success:
                pipeline.log(f"❌ Pipeline stopped at: {name}")
                eel.ui_set_error(f"Failed at {name}: {msg}")()
                return
        except Exception as e:
            pipeline.log(f"❌ Unexpected Error in {name}: {str(e)}")
            eel.ui_set_error(f"Error in {name}: {str(e)}")()
            return

    # Pipeline Complete
    eel.ui_set_complete(pipeline.metrics)()


# ==================== MAIN ====================
def main():
    """Launch the Eel Desktop Application."""
    print("=" * 64)
    print("  Qunart — Hybrid LLM Compression Desktop App")
    print("=" * 64)
    print("\nStarting App UI...")
    
    # Start Eel (opens a Chrome/Edge window in App mode)
    try:
        eel.start('index.html', mode='chrome', size=(1100, 720), port=0)
    except EnvironmentError:
        # Fallback to Edge or default browser if Chrome isn't found
        eel.start('index.html', mode='edge', size=(1100, 720), port=0)

if __name__ == "__main__":
    main()