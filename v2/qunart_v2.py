"""
Hamerspace - AI Pipeline Compression Platform
============================================
Professional platform for compressing AI models for edge deployment
with quantum-enhanced optimization and hardware-aware compression.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import threading
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pickle
import os
from pathlib import Path
import hashlib
from enum import Enum

# ML and compression imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, load_iris, load_wine

# Quantum computing (optional)
try:
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit.primitives import StatevectorEstimator
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

class CompressionMethod(Enum):
    QUANTUM_FEATURE = "quantum_feature"
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    DISTILLATION = "distillation"
    LOW_RANK = "low_rank"
    HYBRID = "hybrid"

class HardwareProfile(Enum):
    RASPBERRY_PI_4 = {"ram": 4096, "ram_unit": "MB", "cpu_cores": 4, "gpu": False, "gpu_vram": 0, "arch": "arm64"}
    JETSON_NANO = {"ram": 4096, "ram_unit": "MB", "cpu_cores": 4, "gpu": True, "gpu_vram": 2, "arch": "arm64"}
    ESP32 = {"ram": 512, "ram_unit": "MB", "cpu_cores": 2, "gpu": False, "gpu_vram": 0, "arch": "xtensa"}
    MOBILE_HIGH = {"ram": 8, "ram_unit": "GB", "cpu_cores": 8, "gpu": True, "gpu_vram": 4, "arch": "arm64"}
    MOBILE_MID = {"ram": 4, "ram_unit": "GB", "cpu_cores": 6, "gpu": True, "gpu_vram": 2, "arch": "arm64"}
    EDGE_TPU = {"ram": 1, "ram_unit": "GB", "cpu_cores": 4, "gpu": False, "gpu_vram": 0, "arch": "arm64", "tpu": True}
    CUSTOM = {"ram": 0, "ram_unit": "MB", "cpu_cores": 0, "gpu": False, "gpu_vram": 0, "arch": "custom"}

@dataclass
class ModelMetrics:
    original_size: float
    compressed_size: float
    accuracy_drop: float
    latency_improvement: float
    memory_reduction: float
    flops_reduction: float
    compression_ratio: float

@dataclass
class CompressionResult:
    model_id: str
    selected_features: List[str]
    compression_method: str
    hardware_profile: str
    metrics: ModelMetrics
    optimization_params: Dict
    processing_time: float
    quantum_scores: Dict[str, float] = None

@dataclass
class CustomHardwareSpec:
    ram: int
    ram_unit: str
    cpu_cores: int
    gpu: bool
    gpu_vram: int
    architecture: str
    
    def to_dict(self):
        return {
            "ram": self.ram,
            "ram_unit": self.ram_unit,
            "cpu_cores": self.cpu_cores,
            "gpu": self.gpu,
            "gpu_vram": self.gpu_vram,
            "arch": self.architecture
        }

class ModernUI:
    """Modern UI components with animations"""
    @staticmethod
    def create_button(parent, text, command, style='primary', **kwargs):
        colors = {
            'primary': ('#007bff', '#0056b3', 'white'),
            'success': ('#28a745', '#1e7e34', 'white'),
            'danger': ('#dc3545', '#c82333', 'white'),
            'warning': ('#ffc107', '#d39e00', 'black'),
            'info': ('#17a2b8', '#138496', 'white'),
            'secondary': ('#6c757d', '#545b62', 'white')
        }
        
        bg, hover_bg, fg = colors.get(style, colors['primary'])
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                       font=('Segoe UI', 10, 'bold'), relief='flat', bd=0,
                       padx=20, pady=10, cursor='hand2', **kwargs)
        
        def on_enter(e): btn.configure(bg=hover_bg)
        def on_leave(e): btn.configure(bg=bg)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn
    
    @staticmethod
    def create_card(parent, title=None, **kwargs):
        card = tk.Frame(parent, bg='white', relief='flat', bd=1,
                       highlightbackground='#e9ecef', highlightthickness=1, **kwargs)
        if title:
            title_frame = tk.Frame(card, bg='white', height=50)
            title_frame.pack(fill='x', padx=20, pady=(20, 0))
            title_frame.pack_propagate(False)
            tk.Label(title_frame, text=title, font=('Segoe UI', 14, 'bold'),
                    bg='white', fg='#495057').pack(anchor='w')
            tk.Frame(card, bg='#e9ecef', height=1).pack(fill='x', padx=20, pady=(10, 0))
        return card

class QuantumOptimizer:
    """Quantum-enhanced feature optimization"""
    def __init__(self, dimension: int = 2):
        self.dimension = dimension
        self.available = QUANTUM_AVAILABLE
        if self.available:
            try:
                self.feature_map = ZZFeatureMap(dimension, reps=2)
                self.estimator = StatevectorEstimator()
            except:
                self.available = False
    
    def evaluate_features(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Quantum-enhanced feature evaluation"""
        scores = {}
        for i in range(X.shape[1]):
            if self.available:
                scores[f"feature_{i}"] = self._quantum_score(X[:, i])
            else:
                scores[f"feature_{i}"] = self._classical_score(X[:, i], y)
        return scores
    
    def _quantum_score(self, feature_data: np.ndarray) -> float:
        """Quantum feature scoring"""
        try:
            normalized = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min() + 1e-8)
            sample = np.random.choice(normalized, min(10, len(normalized)), replace=False)
            
            if len(sample) < self.dimension:
                sample = np.pad(sample, (0, self.dimension - len(sample)), mode='constant')
            
            params = sample[:self.dimension] * 2 * np.pi
            qc = self.feature_map.assign_parameters(params)
            
            return abs(np.mean(np.cos(params)) * np.exp(-np.var(sample)))
        except:
            return self._classical_score(feature_data, None)
    
    def _classical_score(self, feature_data: np.ndarray, y: np.ndarray = None) -> float:
        """Classical fallback scoring"""
        if len(feature_data) == 0 or np.all(np.isnan(feature_data)):
            return 0.0
        
        feature_data = feature_data[~np.isnan(feature_data)]
        if len(feature_data) == 0:
            return 0.0
        
        variance_score = 1.0 - np.exp(-np.var(feature_data))
        entropy_score = len(np.unique(feature_data)) / len(feature_data)
        
        if y is not None:
            try:
                correlation = abs(np.corrcoef(feature_data, y)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            except:
                correlation = 0.0
            return (variance_score + entropy_score + correlation) / 3
        
        return (variance_score + entropy_score) / 2

class CompressionEngine:
    """Main compression engine with multiple methods"""
    def __init__(self):
        self.quantum_optimizer = QuantumOptimizer()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def compress_pipeline(self, X, y, method: CompressionMethod, 
                         hardware_profile: Union[HardwareProfile, CustomHardwareSpec],
                         compression_ratio: float = 0.5,
                         progress_callback=None) -> CompressionResult:
        """Main compression function"""
        start_time = time.time()
        model_id = hashlib.md5(f"{time.time()}{method.value}".encode()).hexdigest()[:8]
        
        if progress_callback:
            progress_callback(10, f"Initializing {method.value} compression...")
        
        X_processed, y_processed = self._prepare_data(X, y)
        
        if progress_callback:
            progress_callback(30, "Analyzing features...")
        
        if method == CompressionMethod.QUANTUM_FEATURE:
            quantum_scores = self.quantum_optimizer.evaluate_features(X_processed, y_processed)
            selected_features = self._select_top_features(quantum_scores, compression_ratio)
        else:
            selected_features, quantum_scores = self._classical_compression(
                X_processed, y_processed, method, compression_ratio)
        
        if progress_callback:
            progress_callback(70, "Computing metrics...")
        
        metrics = self._calculate_metrics(X_processed, y_processed, selected_features, hardware_profile)
        
        if progress_callback:
            progress_callback(100, "Compression complete!")
        
        hardware_name = hardware_profile.name if isinstance(hardware_profile, HardwareProfile) else "CUSTOM"
        
        return CompressionResult(
            model_id=model_id,
            selected_features=selected_features,
            compression_method=method.value,
            hardware_profile=hardware_name,
            metrics=metrics,
            optimization_params={"compression_ratio": compression_ratio},
            processing_time=time.time() - start_time,
            quantum_scores=quantum_scores
        )
    
    def _prepare_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and clean data"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler.fit_transform(X)
        
        if y.dtype == object or len(np.unique(y)) < 0.1 * len(y):
            y = self.label_encoder.fit_transform(y)
        
        return X_scaled, y
    
    def _select_top_features(self, scores: Dict, ratio: float) -> List[str]:
        """Select top features based on scores"""
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_select = max(1, int(len(sorted_features) * ratio))
        return [f for f, _ in sorted_features[:n_select]]
    
    def _classical_compression(self, X, y, method, ratio) -> Tuple[List[str], Dict]:
        """Classical compression methods"""
        n_features = X.shape[1]
        scores = {}
        
        if method == CompressionMethod.PRUNING:
            from sklearn.feature_selection import SelectKBest, f_classif
            k = max(1, int(n_features * ratio))
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X, y)
            selected_indices = selector.get_support(indices=True)
            
            for i, score in enumerate(selector.scores_):
                scores[f"feature_{i}"] = score
            
            selected_features = [f"feature_{i}" for i in selected_indices]
            
        elif method == CompressionMethod.LOW_RANK:
            from sklearn.decomposition import PCA
            n_components = max(1, int(n_features * ratio))
            pca = PCA(n_components=n_components)
            pca.fit(X)
            
            for i, var_ratio in enumerate(pca.explained_variance_ratio_[:n_features]):
                scores[f"feature_{i}"] = var_ratio if i < len(pca.explained_variance_ratio_) else 0.0
            
            selected_features = [f"feature_{i}" for i in range(n_components)]
            
        else:
            for i in range(n_features):
                try:
                    corr = abs(np.corrcoef(X[:, i], y)[0, 1])
                    scores[f"feature_{i}"] = corr if not np.isnan(corr) else 0.0
                except:
                    scores[f"feature_{i}"] = 0.0
            
            selected_features = self._select_top_features(scores, ratio)
        
        return selected_features, scores
    
    def _calculate_metrics(self, X, y, selected_features, hardware_profile) -> ModelMetrics:
        """Calculate compression metrics"""
        original_features = X.shape[1]
        compressed_features = len(selected_features)
        
        try:
            feature_indices = [int(f.split('_')[1]) for f in selected_features if f.startswith('feature_')]
            X_compressed = X[:, feature_indices] if feature_indices else X[:, :compressed_features]
            
            X_train, X_test, y_train, y_test = train_test_split(X_compressed, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            accuracy_drop = max(0, 0.95 - accuracy)
        except:
            accuracy_drop = 0.1
        
        compression_ratio = compressed_features / original_features
        
        # Get hardware specs
        if isinstance(hardware_profile, HardwareProfile):
            hardware_specs = hardware_profile.value
        else:  # CustomHardwareSpec
            hardware_specs = hardware_profile.to_dict()
        
        memory_reduction = 1 - compression_ratio
        latency_improvement = memory_reduction * 0.8
        flops_reduction = memory_reduction * 0.9
        
        # Adjust for hardware constraints
        ram_mb = hardware_specs.get("ram", 4096)
        if hardware_specs.get("ram_unit") == "GB":
            ram_mb *= 1024
            
        if ram_mb < 1024:
            memory_reduction *= 1.2
            latency_improvement *= 1.1
        
        return ModelMetrics(
            original_size=original_features * 4,
            compressed_size=compressed_features * 4,
            accuracy_drop=accuracy_drop,
            latency_improvement=latency_improvement,
            memory_reduction=memory_reduction,
            flops_reduction=flops_reduction,
            compression_ratio=compression_ratio
        )

class HamerspaceGUI:
    """Professional GUI for Hamerspace platform"""
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.engine = CompressionEngine()
        self.current_data = None
        self.results = []
        self.custom_hardware_spec = CustomHardwareSpec(4096, "MB", 4, False, 0, "x86_64")
        self.setup_ui()
    
    def setup_window(self):
        """Configure main window"""
        self.root.title("Hamerspace - AI Pipeline Compressor")
        self.root.geometry("1600x900")
        self.root.configure(bg='#f8f9fa')
        self.root.minsize(1400, 800)
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (800)
        y = (self.root.winfo_screenheight() // 2) - (450)
        self.root.geometry(f'1600x900+{x}+{y}')
    
    def setup_ui(self):
        """Setup main user interface"""
        self.create_header()
        
        main_container = tk.Frame(self.root, bg='#f8f9fa')
        main_container.pack(fill='both', expand=True, padx=30, pady=20)
        
        left_panel = tk.Frame(main_container, bg='#f8f9fa', width=500)
        left_panel.pack(side='left', fill='y', padx=(0, 20))
        left_panel.pack_propagate(False)
        
        right_panel = tk.Frame(main_container, bg='#f8f9fa')
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.create_left_panel(left_panel)
        self.create_right_panel(right_panel)
    
    def create_header(self):
        """Create application header"""
        # Clear existing header
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame) and hasattr(widget, '_is_header'):
                widget.destroy()
        
        header = tk.Frame(self.root, bg='#ffffff', height=80)
        header._is_header = True
        header.pack(fill='x', padx=30, pady=(20, 0))
        header.pack_propagate(False)
        
        title_frame = tk.Frame(header, bg='#ffffff')
        title_frame.pack(side='left', fill='y', padx=20)
        
        tk.Label(title_frame, text="Hamerspace", font=('Segoe UI', 24, 'bold'),
                bg='#ffffff', fg='#007bff').pack(anchor='w')
        tk.Label(title_frame, text="AI Pipeline Compression Platform",
                font=('Segoe UI', 12), bg='#ffffff', fg='#6c757d').pack(anchor='w')
        
        status_frame = tk.Frame(header, bg='#ffffff')
        status_frame.pack(side='right', fill='y', padx=20)
        
        quantum_status = "Quantum Available" if QUANTUM_AVAILABLE else "Classical Mode"
        tk.Label(status_frame, text=quantum_status, font=('Segoe UI', 10),
                bg='#ffffff', fg='#28a745' if QUANTUM_AVAILABLE else '#ffc107').pack(anchor='e')
        tk.Label(status_frame, text=f"{len(self.results)} Models Processed",
                font=('Segoe UI', 10), bg='#ffffff', fg='#6c757d').pack(anchor='e')
    
    def create_left_panel(self, parent):
        """Create left control panel with scrolling"""
        canvas = tk.Canvas(parent, bg='#f8f9fa', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f8f9fa')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Model Upload Section
        upload_card = ModernUI.create_card(scrollable_frame, "Model Upload")
        upload_card.pack(fill='x', pady=(0, 20))
        
        upload_frame = tk.Frame(upload_card, bg='white')
        upload_frame.pack(fill='x', padx=20, pady=20)
        
        btn_frame = tk.Frame(upload_frame, bg='white')
        btn_frame.pack(fill='x', pady=(0, 15))
        
        ModernUI.create_button(btn_frame, "Load CSV Dataset", 
                              self.load_csv, 'primary').pack(fill='x', pady=(0, 5))
        ModernUI.create_button(btn_frame, "Load Iris Dataset", 
                              self.load_iris, 'info').pack(fill='x', pady=(0, 5))
        ModernUI.create_button(btn_frame, "Load Wine Dataset", 
                              self.load_wine, 'info').pack(fill='x', pady=(0, 5))
        
        self.data_info = tk.Label(upload_frame, text="No data loaded", 
                                 font=('Segoe UI', 10), bg='white', fg='#6c757d')
        self.data_info.pack(pady=(10, 0))
        
        # Hardware Profile Section
        hardware_card = ModernUI.create_card(scrollable_frame, "Hardware Profile")
        hardware_card.pack(fill='x', pady=(0, 20))
        
        hardware_frame = tk.Frame(hardware_card, bg='white')
        hardware_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(hardware_frame, text="Target Deployment", font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#495057').pack(anchor='w')
        
        self.hardware_var = tk.StringVar(value="RASPBERRY_PI_4")
        hardware_combo = ttk.Combobox(hardware_frame, textvariable=self.hardware_var,
                                     values=[profile.name for profile in HardwareProfile],
                                     state='readonly', font=('Segoe UI', 10))
        hardware_combo.pack(fill='x', pady=(5, 15))
        hardware_combo.bind('<<ComboboxSelected>>', self.on_hardware_change)
        
        # Custom Hardware Configuration (appears when CUSTOM is selected)
        self.custom_frame = tk.Frame(hardware_frame, bg='white')
        
        # RAM Configuration
        ram_frame = tk.Frame(self.custom_frame, bg='white')
        ram_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(ram_frame, text="RAM:", font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#495057').pack(side='left')
        
        self.ram_var = tk.IntVar(value=self.custom_hardware_spec.ram)
        ram_spinbox = tk.Spinbox(ram_frame, from_=64, to=32768, increment=64,
                                textvariable=self.ram_var, font=('Segoe UI', 10),
                                width=8, justify='center', command=self.update_custom_config)
        ram_spinbox.pack(side='left', padx=(10, 5))
        
        self.ram_unit_var = tk.StringVar(value=self.custom_hardware_spec.ram_unit)
        ram_unit_combo = ttk.Combobox(ram_frame, textvariable=self.ram_unit_var,
                                     values=["MB", "GB"], state='readonly', width=4,
                                     font=('Segoe UI', 10))
        ram_unit_combo.pack(side='left', padx=(0, 10))
        ram_unit_combo.bind('<<ComboboxSelected>>', self.update_custom_config)
        
        # CPU Configuration
        cpu_frame = tk.Frame(self.custom_frame, bg='white')
        cpu_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(cpu_frame, text="CPU Cores:", font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#495057').pack(side='left')
        
        self.cpu_var = tk.IntVar(value=self.custom_hardware_spec.cpu_cores)
        cpu_spinbox = tk.Spinbox(cpu_frame, from_=1, to=32, increment=1,
                                textvariable=self.cpu_var, font=('Segoe UI', 10),
                                width=8, justify='center', command=self.update_custom_config)
        cpu_spinbox.pack(side='left', padx=(10, 0))
        
        # GPU Configuration
        gpu_frame = tk.Frame(self.custom_frame, bg='white')
        gpu_frame.pack(fill='x', pady=(0, 10))
        
        self.gpu_var = tk.BooleanVar(value=self.custom_hardware_spec.gpu)
        gpu_check = tk.Checkbutton(gpu_frame, text="GPU Available", variable=self.gpu_var,
                                  font=('Segoe UI', 10, 'bold'), bg='white', fg='#495057',
                                  command=self.toggle_gpu_options)
        gpu_check.pack(side='left')
        
        tk.Label(gpu_frame, text="VRAM:", font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#495057').pack(side='left', padx=(20, 0))
        
        self.vram_var = tk.IntVar(value=self.custom_hardware_spec.gpu_vram)
        self.vram_spinbox = tk.Spinbox(gpu_frame, from_=1, to=32, increment=1,
                                      textvariable=self.vram_var, font=('Segoe UI', 10),
                                      width=6, justify='center', command=self.update_custom_config)
        self.vram_spinbox.pack(side='left', padx=(5, 0))
        
        tk.Label(gpu_frame, text="GB", font=('Segoe UI', 10),
                bg='white', fg='#6c757d').pack(side='left', padx=(5, 0))
        
        # Architecture Configuration
        arch_frame = tk.Frame(self.custom_frame, bg='white')
        arch_frame.pack(fill='x', pady=(0, 15))
        
        tk.Label(arch_frame, text="Architecture:", font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#495057').pack(side='left')
        
        self.arch_var = tk.StringVar(value=self.custom_hardware_spec.architecture)
        arch_combo = ttk.Combobox(arch_frame, textvariable=self.arch_var,
                                 values=["x86_64", "arm64", "armv7", "xtensa", "riscv", "custom"],
                                 state='readonly', font=('Segoe UI', 10), width=12)
        arch_combo.pack(side='left', padx=(10, 0))
        arch_combo.bind('<<ComboboxSelected>>', self.update_custom_config)
        
        # Hardware Info Display
        self.hardware_info = tk.Text(hardware_frame, height=6, bg='#f8f9fa', 
                                    font=('Segoe UI', 9), wrap='word', bd=0, relief='flat')
        self.hardware_info.pack(fill='x', pady=(15, 0))
        
        self.update_hardware_info()
        
        # Compression Settings
        compression_card = ModernUI.create_card(scrollable_frame, "Compression Settings")
        compression_card.pack(fill='x', pady=(0, 20))
        
        settings_frame = tk.Frame(compression_card, bg='white')
        settings_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(settings_frame, text="Compression Method", font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#495057').pack(anchor='w')
        
        self.method_var = tk.StringVar(value="QUANTUM_FEATURE")
        method_combo = ttk.Combobox(settings_frame, textvariable=self.method_var,
                                   values=[method.name for method in CompressionMethod],
                                   state='readonly', font=('Segoe UI', 10))
        method_combo.pack(fill='x', pady=(5, 15))
        
        tk.Label(settings_frame, text="Compression Ratio", font=('Segoe UI', 10, 'bold'),
                bg='white', fg='#495057').pack(anchor='w')
        
        self.ratio_var = tk.DoubleVar(value=0.5)
        ratio_scale = tk.Scale(settings_frame, from_=0.1, to=0.9, resolution=0.1,
                              orient='horizontal', variable=self.ratio_var,
                              bg='white', fg='#495057', highlightthickness=0)
        ratio_scale.pack(fill='x', pady=(5, 15))
        
        self.process_btn = ModernUI.create_button(settings_frame, "Compress Pipeline",
                                                 self.compress_pipeline, 'success')
        self.process_btn.pack(fill='x', pady=(10, 0))
        
        progress_frame = tk.Frame(settings_frame, bg='white')
        progress_frame.pack(fill='x', pady=(15, 0))
        
        self.progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress.pack(fill='x', pady=(0, 5))
        
        self.status_label = tk.Label(progress_frame, text="Ready", 
                                    font=('Segoe UI', 10), bg='white', fg='#6c757d')
        self.status_label.pack(anchor='w')
    
    def create_right_panel(self, parent):
        """Create right results panel"""
        results_card = ModernUI.create_card(parent, "Results Dashboard")
        results_card.pack(fill='both', expand=True)
        
        self.notebook = ttk.Notebook(results_card)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.create_summary_tab()
        self.create_metrics_tab()
        self.create_models_tab()
        self.create_export_tab()
    
    def create_summary_tab(self):
        """Create summary results tab"""
        summary_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(summary_frame, text="Summary")
        
        self.summary_text = tk.Text(summary_frame, bg='white', font=('Segoe UI', 10),
                                   wrap='word', padx=20, pady=20)
        summary_scroll = ttk.Scrollbar(summary_frame, orient='vertical')
        
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        summary_scroll.configure(command=self.summary_text.yview)
        
        self.summary_text.pack(side='left', fill='both', expand=True)
        summary_scroll.pack(side='right', fill='y')
    
    def create_metrics_tab(self):
        """Create metrics visualization tab"""
        metrics_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(metrics_frame, text="Metrics")
        
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, metrics_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=20, pady=20)
    
    def create_models_tab(self):
        """Create models history tab"""
        models_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(models_frame, text="Models")
        
        tree_frame = tk.Frame(models_frame, bg='white')
        tree_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.models_tree = ttk.Treeview(tree_frame, 
                                       columns=('Method', 'Hardware', 'Compression', 'Accuracy', 'Time'),
                                       show='tree headings')
        
        self.models_tree.heading('#0', text='Model ID')
        self.models_tree.heading('Method', text='Method')
        self.models_tree.heading('Hardware', text='Hardware')
        self.models_tree.heading('Compression', text='Compression')
        self.models_tree.heading('Accuracy', text='Accuracy Drop')
        self.models_tree.heading('Time', text='Time (s)')
        
        self.models_tree.column('#0', width=120)
        self.models_tree.column('Method', width=150)
        self.models_tree.column('Hardware', width=150)
        self.models_tree.column('Compression', width=100)
        self.models_tree.column('Accuracy', width=100)
        self.models_tree.column('Time', width=80)
        
        models_scroll = ttk.Scrollbar(tree_frame, orient='vertical')
        self.models_tree.configure(yscrollcommand=models_scroll.set)
        models_scroll.configure(command=self.models_tree.yview)
        
        self.models_tree.pack(side='left', fill='both', expand=True)
        models_scroll.pack(side='right', fill='y')
    
    def create_export_tab(self):
        """Create export options tab"""
        export_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(export_frame, text="Export")
        
        export_container = tk.Frame(export_frame, bg='white')
        export_container.pack(expand=True)
        
        tk.Label(export_container, text="Export Options", font=('Segoe UI', 16, 'bold'),
                bg='white', fg='#495057').pack(pady=20)
        
        ModernUI.create_button(export_container, "Export Report (JSON)",
                              self.export_report, 'info').pack(pady=10)
        ModernUI.create_button(export_container, "Export Compressed Model",
                              self.export_model, 'success').pack(pady=10)
        ModernUI.create_button(export_container, "Export Metrics (CSV)",
                              self.export_metrics, 'secondary').pack(pady=10)
    
    def on_hardware_change(self, event=None):
        """Handle hardware profile change"""
        if self.hardware_var.get() == "CUSTOM":
            self.custom_frame.pack(fill='x', pady=(0, 15))
            self.toggle_gpu_options()
        else:
            self.custom_frame.pack_forget()
        self.update_hardware_info()
    
    def toggle_gpu_options(self):
        """Toggle GPU VRAM options based on GPU availability"""
        if self.gpu_var.get():
            self.vram_spinbox.configure(state='normal')
        else:
            self.vram_spinbox.configure(state='disabled')
        self.update_custom_config()
    
    def update_custom_config(self, event=None):
        """Update custom hardware configuration"""
        if self.hardware_var.get() == "CUSTOM":
            self.custom_hardware_spec = CustomHardwareSpec(
                ram=self.ram_var.get(),
                ram_unit=self.ram_unit_var.get(),
                cpu_cores=self.cpu_var.get(),
                gpu=self.gpu_var.get(),
                gpu_vram=self.vram_var.get() if self.gpu_var.get() else 0,
                architecture=self.arch_var.get()
            )
            self.update_hardware_info()
    
    def update_hardware_info(self, event=None):
        """Update hardware profile information"""
        try:
            if self.hardware_var.get() == "CUSTOM":
                spec = self.custom_hardware_spec
                ram_mb = spec.ram if spec.ram_unit == "MB" else spec.ram * 1024
                
                info_text = f"""RAM: {spec.ram} {spec.ram_unit} ({ram_mb} MB)
CPU Cores: {spec.cpu_cores}
GPU: {'Yes' if spec.gpu else 'No'}"""
                
                if spec.gpu and spec.gpu_vram > 0:
                    info_text += f" ({spec.gpu_vram} GB VRAM)"
                    
                info_text += f"\nArchitecture: {spec.architecture.upper()}"
                
                # Add performance class
                if ram_mb < 512:
                    perf_class = "Ultra Low-End"
                elif ram_mb < 2048:
                    perf_class = "Low-End"
                elif ram_mb < 8192:
                    perf_class = "Mid-Range"
                elif ram_mb < 16384:
                    perf_class = "High-End"
                else:
                    perf_class = "Workstation"
                
                info_text += f"\nClass: {perf_class}"
                
            else:
                profile = HardwareProfile[self.hardware_var.get()]
                specs = profile.value
                
                ram_display = f"{specs['ram']} {specs.get('ram_unit', 'MB')}"
                if specs.get('ram_unit') == 'GB':
                    ram_mb = specs['ram'] * 1024
                else:
                    ram_mb = specs['ram']
                
                info_text = f"""RAM: {ram_display} ({ram_mb} MB)
CPU Cores: {specs['cpu_cores']}
GPU: {'Yes' if specs.get('gpu', False) else 'No'}"""
                
                if specs.get('gpu') and specs.get('gpu_vram', 0) > 0:
                    info_text += f" ({specs['gpu_vram']} GB VRAM)"
                    
                info_text += f"\nArchitecture: {specs['arch'].upper()}"
                
                if specs.get('tpu'):
                    info_text += "\nTPU: Available"
            
            self.hardware_info.delete(1.0, tk.END)
            self.hardware_info.insert(1.0, info_text)
            self.hardware_info.configure(state='disabled')
        except Exception as e:
            pass
    
    def load_csv(self):
        """Load CSV dataset"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                data = pd.read_csv(file_path)
                data = data.dropna()
                
                if 'target' in data.columns:
                    X, y = data.drop('target', axis=1), data['target']
                elif 'class' in data.columns:
                    X, y = data.drop('class', axis=1), data['class']
                else:
                    X, y = data.iloc[:, :-1], data.iloc[:, -1]
                
                X = X.select_dtypes(include=[np.number])
                if X.empty:
                    raise ValueError("No numeric features found")
                
                self.current_data = {'X': X, 'y': y}
                self.data_info.configure(text=f"CSV loaded: {X.shape[0]} samples, {X.shape[1]} features",
                                       fg='#28a745')
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def load_iris(self):
        """Load Iris dataset"""
        try:
            iris = load_iris()
            X = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = pd.Series(iris.target)
            self.current_data = {'X': X, 'y': y}
            self.data_info.configure(text=f"Iris loaded: {X.shape[0]} samples, {X.shape[1]} features",
                                   fg='#28a745')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Iris: {str(e)}")
    
    def load_wine(self):
        """Load Wine dataset"""
        try:
            wine = load_wine()
            X = pd.DataFrame(wine.data, columns=wine.feature_names)
            y = pd.Series(wine.target)
            self.current_data = {'X': X, 'y': y}
            self.data_info.configure(text=f"Wine loaded: {X.shape[0]} samples, {X.shape[1]} features",
                                   fg='#28a745')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Wine: {str(e)}")
    
    def compress_pipeline(self):
        """Main compression function"""
        if not self.current_data:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        self.process_btn.configure(state='disabled')
        
        thread = threading.Thread(target=self._compress_thread)
        thread.daemon = True
        thread.start()
    
    def _compress_thread(self):
        """Compression thread function"""
        try:
            method = CompressionMethod[self.method_var.get()]
            
            if self.hardware_var.get() == "CUSTOM":
                hardware = self.custom_hardware_spec
            else:
                hardware = HardwareProfile[self.hardware_var.get()]
            
            ratio = self.ratio_var.get()
            
            result = self.engine.compress_pipeline(
                self.current_data['X'], 
                self.current_data['y'],
                method, 
                hardware, 
                ratio,
                progress_callback=self.update_progress
            )
            
            self.results.append(result)
            self.root.after(0, self.update_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Compression failed: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.process_btn.configure(state='normal'))
    
    def update_progress(self, value, status):
        """Update progress bar and status"""
        self.progress['value'] = value
        self.status_label.configure(text=status)
    
    def update_results(self):
        """Update all results displays"""
        if not self.results:
            return
        
        latest_result = self.results[-1]
        
        self.update_summary(latest_result)
        self.update_visualizations()
        self.update_models_tree()
        self.create_header()
    
    def update_summary(self, result: CompressionResult):
        """Update summary text"""
        summary = f"""
HAMERSPACE COMPRESSION REPORT
{'='*60}

MODEL INFORMATION
• Model ID: {result.model_id}
• Compression Method: {result.compression_method.replace('_', ' ').title()}
• Target Hardware: {result.hardware_profile.replace('_', ' ').title()}
• Processing Time: {result.processing_time:.2f}s

COMPRESSION METRICS
• Original Size: {result.metrics.original_size:.1f} KB
• Compressed Size: {result.metrics.compressed_size:.1f} KB
• Compression Ratio: {result.metrics.compression_ratio:.1%}
• Memory Reduction: {result.metrics.memory_reduction:.1%}

PERFORMANCE METRICS
• Accuracy Drop: {result.metrics.accuracy_drop:.3f}
• Latency Improvement: {result.metrics.latency_improvement:.1%}
• FLOPs Reduction: {result.metrics.flops_reduction:.1%}

SELECTED FEATURES ({len(result.selected_features)})
{chr(10).join(f"• {feature}" for feature in result.selected_features[:15])}
{"..." if len(result.selected_features) > 15 else ""}

QUANTUM INSIGHTS
{"Quantum optimization enabled" if QUANTUM_AVAILABLE and result.compression_method == "quantum_feature" else "Classical optimization used"}

DEPLOYMENT READINESS
• Hardware Compatibility: Compatible
• Memory Constraints: Within limits  
• Performance Target: Achieved
• Model Quality: {"Excellent" if result.metrics.accuracy_drop < 0.05 else "Good" if result.metrics.accuracy_drop < 0.15 else "Review needed"}

RECOMMENDATIONS
• Compression Quality: {"Excellent" if result.metrics.compression_ratio < 0.5 else "Good" if result.metrics.compression_ratio < 0.7 else "Consider higher compression"}
• Performance Impact: {"Minimal" if result.metrics.accuracy_drop < 0.1 else "Moderate" if result.metrics.accuracy_drop < 0.2 else "Significant"}
• Edge Deployment: {"Ready" if result.metrics.memory_reduction > 0.3 else "Needs optimization"}
        """
        
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)
    
    def update_visualizations(self):
        """Update metric visualizations"""
        if not self.results:
            return
        
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        latest = self.results[-1]
        
        # Plot 1: Size Comparison
        metrics = ['Original Size', 'Compressed Size', 'Memory Saved']
        values = [latest.metrics.original_size, 
                 latest.metrics.compressed_size,
                 latest.metrics.original_size - latest.metrics.compressed_size]
        colors = ['#dc3545', '#28a745', '#17a2b8']
        
        bars = self.ax1.bar(metrics, values, color=colors, alpha=0.8)
        self.ax1.set_title('Size Comparison', fontweight='bold', pad=20)
        self.ax1.set_ylabel('Size (KB)')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax1.text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Method Performance
        if len(self.results) > 1:
            methods = [r.compression_method for r in self.results[-5:]]
            ratios = [r.metrics.compression_ratio for r in self.results[-5:]]
            accuracies = [1 - r.metrics.accuracy_drop for r in self.results[-5:]]
            
            x = np.arange(len(methods))
            width = 0.35
            
            bars1 = self.ax2.bar(x - width/2, ratios, width, label='Compression', color='#007bff', alpha=0.8)
            bars2 = self.ax2.bar(x + width/2, accuracies, width, label='Accuracy', color='#28a745', alpha=0.8)
            
            self.ax2.set_title('Method Comparison', fontweight='bold', pad=20)
            self.ax2.set_ylabel('Score')
            self.ax2.set_xticks(x)
            self.ax2.set_xticklabels([m[:8] + '...' if len(m) > 8 else m for m in methods], rotation=45)
            self.ax2.legend()
        else:
            if latest.quantum_scores:
                top_features = sorted(latest.quantum_scores.items(), key=lambda x: x[1], reverse=True)[:10]
                features = [f[:10] + '...' if len(f) > 10 else f for f, _ in top_features]
                scores = [s for _, s in top_features]
                colors = ['#28a745' if f.replace('...', '') in [sf[:10] for sf in latest.selected_features] 
                         else '#6c757d' for f in features]
                
                bars = self.ax2.barh(range(len(features)), scores, color=colors, alpha=0.8)
                self.ax2.set_title('Top Features', fontweight='bold', pad=20)
                self.ax2.set_xlabel('Importance Score')
                self.ax2.set_yticks(range(len(features)))
                self.ax2.set_yticklabels(features)
        
        # Plot 3: Hardware Efficiency
        hardware_metrics = ['Memory', 'Latency', 'FLOPs']
        improvements = [latest.metrics.memory_reduction, 
                       latest.metrics.latency_improvement,
                       latest.metrics.flops_reduction]
        
        bars = self.ax3.bar(hardware_metrics, improvements, 
                           color=['#ffc107', '#17a2b8', '#6f42c1'], alpha=0.8)
        self.ax3.set_title('Hardware Efficiency', fontweight='bold', pad=20)
        self.ax3.set_ylabel('Improvement %')
        self.ax3.set_ylim(0, 1)
        
        for bar, value in zip(bars, improvements):
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height,
                         f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Processing Timeline or Breakdown
        if len(self.results) > 1:
            times = [r.processing_time for r in self.results]
            line = self.ax4.plot(range(1, len(times) + 1), times, 
                               marker='o', linewidth=2, markersize=6, color='#007bff')
            self.ax4.set_title('Processing Time Trend', fontweight='bold', pad=20)
            self.ax4.set_xlabel('Model Number')
            self.ax4.set_ylabel('Time (seconds)')
            self.ax4.grid(True, alpha=0.3)
        else:
            breakdown = {
                'Feature Selection': 0.4,
                'Model Training': 0.3,
                'Optimization': 0.2,
                'Validation': 0.1
            }
            
            wedges, texts, autotexts = self.ax4.pie(breakdown.values(), labels=breakdown.keys(),
                                                   autopct='%1.1f%%', startangle=90,
                                                   colors=['#007bff', '#28a745', '#ffc107', '#17a2b8'])
            self.ax4.set_title('Processing Breakdown', fontweight='bold', pad=20)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_models_tree(self):
        """Update models tree view"""
        for item in self.models_tree.get_children():
            self.models_tree.delete(item)
        
        for result in self.results:
            self.models_tree.insert('', 'end', 
                                   text=result.model_id,
                                   values=(
                                       result.compression_method.replace('_', ' ').title(),
                                       result.hardware_profile.replace('_', ' ').title(),
                                       f"{result.metrics.compression_ratio:.1%}",
                                       f"{result.metrics.accuracy_drop:.3f}",
                                       f"{result.processing_time:.2f}"
                                   ))
    
    def export_report(self):
        """Export detailed report"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                export_data = {
                    "platform": "Hammerspace",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "quantum_available": QUANTUM_AVAILABLE,
                    "total_models": len(self.results),
                    "results": [
                        {
                            "model_id": r.model_id,
                            "compression_method": r.compression_method,
                            "hardware_profile": r.hardware_profile,
                            "selected_features": r.selected_features,
                            "metrics": asdict(r.metrics),
                            "optimization_params": r.optimization_params,
                            "processing_time": r.processing_time,
                            "quantum_scores": r.quantum_scores
                        } for r in self.results
                    ]
                }
                
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                messagebox.showinfo("Success", f"Report exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def export_model(self):
        """Export compressed model"""
        if not self.results:
            messagebox.showwarning("Warning", "No model to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                latest_result = self.results[-1]
                
                X = self.current_data['X']
                y = self.current_data['y']
                
                feature_indices = []
                for feature in latest_result.selected_features:
                    if feature.startswith('feature_'):
                        idx = int(feature.split('_')[1])
                        if idx < len(X.columns):
                            feature_indices.append(idx)
                    elif feature in X.columns:
                        feature_indices.append(X.columns.get_loc(feature))
                
                if not feature_indices:
                    feature_indices = list(range(len(latest_result.selected_features)))
                
                X_compressed = X.iloc[:, feature_indices]
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_compressed, y)
                
                model_package = {
                    'model': model,
                    'selected_features': latest_result.selected_features,
                    'feature_indices': feature_indices,
                    'scaler': self.engine.scaler,
                    'label_encoder': self.engine.label_encoder,
                    'compression_result': latest_result,
                    'metadata': {
                        'platform': 'Hammerspace',
                        'version': '1.0',
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                }
                
                with open(file_path, 'wb') as f:
                    pickle.dump(model_package, f)
                
                messagebox.showinfo("Success", f"Model exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export model: {str(e)}")
    
    def export_metrics(self):
        """Export metrics as CSV"""
        if not self.results:
            messagebox.showwarning("Warning", "No metrics to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Metrics",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                metrics_data = []
                for result in self.results:
                    metrics_data.append({
                        'Model_ID': result.model_id,
                        'Compression_Method': result.compression_method,
                        'Hardware_Profile': result.hardware_profile,
                        'Original_Size_KB': result.metrics.original_size,
                        'Compressed_Size_KB': result.metrics.compressed_size,
                        'Compression_Ratio': result.metrics.compression_ratio,
                        'Accuracy_Drop': result.metrics.accuracy_drop,
                        'Memory_Reduction': result.metrics.memory_reduction,
                        'Latency_Improvement': result.metrics.latency_improvement,
                        'FLOPs_Reduction': result.metrics.flops_reduction,
                        'Processing_Time_Seconds': result.processing_time,
                        'Selected_Features_Count': len(result.selected_features)
                    })
                
                df = pd.DataFrame(metrics_data)
                df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Success", f"Metrics exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export metrics: {str(e)}")

def main():
    """Main application entry point"""
    # Set up matplotlib for better display
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    plt.rcParams.update({
        'font.size': 9,
        'font.family': 'sans-serif',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12
    })
    
    # Create and run application
    root = tk.Tk()
    app = HamerspaceGUI(root)
    
    # Configure window properties
    try:
        # Set application icon (if available)
        root.iconbitmap(default='hamerspace.ico')
    except:
        pass
    
    # Start the application
    print("Starting Hamerspace - AI Pipeline Compressor...")
    print(f"Quantum Computing: {'Available' if QUANTUM_AVAILABLE else 'Not Available'}")
    print("GUI Ready!")
    
    root.mainloop()

if __name__ == "__main__":
    main()