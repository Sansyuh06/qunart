"""
Qunart v2 — Professional Desktop GUI

A modern, thread-safe graphical interface for the Qunart LLM compression
framework. Uses CustomTkinter for a native-looking dark application.

Run with:
    python gui.py
"""

import os
import re
import sys
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from datetime import datetime

import customtkinter as ctk

from qunart import CompressionTarget, CompressionPipeline


# ---------------------------------------------------------------------------
# Appearance
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ---------------------------------------------------------------------------
# Thread-safe stdout capture
# ---------------------------------------------------------------------------
class QueueStdout:
    """Redirects stdout into a queue so the GUI can read it safely."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text: str):
        if text and text.strip():
            self.q.put(text)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------
class QunartApp(ctk.CTk):
    """Professional GUI for Qunart LLM compression."""

    def __init__(self):
        super().__init__()

        self.title("Qunart v2 — LLM Compression Framework")
        self.geometry("1280x820")
        self.minsize(1100, 720)

        # State
        self.log_queue: queue.Queue = queue.Queue()
        self.compression_thread: threading.Thread | None = None
        self.is_running: bool = False
        self.old_stdout = sys.stdout

        # Stages used by the pipeline (matches pipeline.py print markers)
        self.stages = [
            "Loading Model",
            "Planning Compression",
            "Pruning",
            "Recovery Fine-Tuning",
            "Exporting Model",
        ]
        self.stage_labels: list[ctk.CTkLabel] = []
        self.stage_bars: list[ctk.CTkProgressBar] = []

        self._build_layout()
        self._show_view("dashboard")
        self._start_queue_polling()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    def _build_layout(self):
        """Create the sidebar and main content area."""
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self._build_sidebar()

        # Main container
        self.main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        # Views
        self.views: dict[str, ctk.CTkFrame] = {}
        for name in ("dashboard", "compress", "progress", "about"):
            frame = ctk.CTkFrame(self.main, corner_radius=16)
            self.views[name] = frame

        self._build_dashboard(self.views["dashboard"])
        self._build_compress(self.views["compress"])
        self._build_progress(self.views["progress"])
        self._build_about(self.views["about"])

    def _build_sidebar(self):
        """Create the branded sidebar with navigation."""
        brand = ctk.CTkLabel(
            self.sidebar,
            text="QUNART",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=("#1f6aa5", "#5dade2"),
        )
        brand.pack(pady=(35, 5))

        tagline = ctk.CTkLabel(
            self.sidebar,
            text="Universal LLM Compression",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
        )
        tagline.pack(pady=(0, 45))

        nav_items = [
            ("🏠  Dashboard", "dashboard"),
            ("🗜️  Compress", "compress"),
            ("📊  Progress", "progress"),
            ("ℹ️  About", "about"),
        ]

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        for label, view in nav_items:
            btn = ctk.CTkButton(
                self.sidebar,
                text=label,
                width=200,
                height=44,
                corner_radius=10,
                font=ctk.CTkFont(size=14),
                command=lambda v=view: self._show_view(v),
            )
            btn.pack(pady=6)
            self.nav_buttons[view] = btn

        # Status footer
        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Ready",
            font=ctk.CTkFont(size=12),
            text_color="gray50",
        )
        self.status_label.pack(side="bottom", pady=25)

    def _show_view(self, name: str):
        """Switch main view and highlight the active nav button."""
        for frame in self.views.values():
            frame.pack_forget()

        view = self.views[name]
        view.pack(fill="both", expand=True)

        for view_name, btn in self.nav_buttons.items():
            if view_name == name:
                btn.configure(fg_color=("#3b8ed0", "#1f6aa5"), hover_color=("#36719f", "#144870"))
            else:
                btn.configure(fg_color=("#3a7ebf", "#1f538d"), hover_color=("#325882", "#14375e"))

    # -----------------------------------------------------------------------
    # Dashboard view
    # -----------------------------------------------------------------------
    def _build_dashboard(self, parent: ctk.CTkFrame):
        header = ctk.CTkLabel(
            parent,
            text="Welcome to Qunart",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        header.pack(anchor="w", padx=35, pady=(35, 10))

        sub = ctk.CTkLabel(
            parent,
            text="Compress any Hugging Face LLM to a target parameter budget with minimal quality loss.",
            font=ctk.CTkFont(size=15),
            text_color="gray70",
            wraplength=750,
            justify="left",
        )
        sub.pack(anchor="w", padx=35, pady=(0, 30))

        cards_frame = ctk.CTkFrame(parent, fg_color="transparent")
        cards_frame.pack(fill="x", padx=35, pady=10)
        cards_frame.grid_columnconfigure((0, 1, 2), weight=1)

        cards = [
            ("Upload Model", "Load any Hugging Face checkpoint or local safetensors model."),
            ("Set Target", "Specify target parameters or target size in GB."),
            ("Compress & Export", "Prune, recover with LoRA, and export a smaller model."),
        ]

        for i, (title, desc) in enumerate(cards):
            card = ctk.CTkFrame(cards_frame, corner_radius=12, height=140)
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            card.grid_propagate(False)

            ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(size=17, weight="bold"),
            ).pack(anchor="w", padx=20, pady=(20, 8))

            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(size=13),
                text_color="gray70",
                wraplength=260,
                justify="left",
            ).pack(anchor="w", padx=20, pady=(0, 15))

        start_btn = ctk.CTkButton(
            parent,
            text="Start Compressing →",
            width=220,
            height=50,
            corner_radius=10,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=lambda: self._show_view("compress"),
        )
        start_btn.pack(anchor="w", padx=35, pady=(30, 10))

        supported = ctk.CTkLabel(
            parent,
            text="Supported families: Llama / Qwen2 / Mistral / Phi-3",
            font=ctk.CTkFont(size=13),
            text_color="gray60",
        )
        supported.pack(anchor="w", padx=35, pady=(10, 0))

    # -----------------------------------------------------------------------
    # Compress view
    # -----------------------------------------------------------------------
    def _build_compress(self, parent: ctk.CTkFrame):
        header = ctk.CTkLabel(
            parent,
            text="Compress a Model",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        header.pack(anchor="w", padx=30, pady=(25, 5))

        sub = ctk.CTkLabel(
            parent,
            text="Configure the source model, target size, and recovery options.",
            font=ctk.CTkFont(size=13),
            text_color="gray70",
        )
        sub.pack(anchor="w", padx=30, pady=(0, 20))

        # Scrollable form area
        scroll = ctk.CTkScrollableFrame(parent, corner_radius=12)
        scroll.pack(fill="both", expand=True, padx=30, pady=(0, 20))

        # --- Model path ---
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=12)
        ctk.CTkLabel(row, text="Model Path / HF ID", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(
            row,
            text="e.g. meta-llama/Llama-2-7b-hf or a local folder path",
            font=ctk.CTkFont(size=11),
            text_color="gray60",
        ).pack(anchor="w", pady=(2, 6))
        model_row = ctk.CTkFrame(row, fg_color="transparent")
        model_row.pack(fill="x")
        self.model_entry = ctk.CTkEntry(
            model_row,
            placeholder_text="meta-llama/Llama-2-7b-hf",
            height=40,
            font=ctk.CTkFont(size=13),
        )
        self.model_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(
            model_row,
            text="Browse",
            width=100,
            height=40,
            command=self._browse_model,
        ).pack(side="right")

        # --- Output dir ---
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=12)
        ctk.CTkLabel(row, text="Output Directory", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        out_row = ctk.CTkFrame(row, fg_color="transparent")
        out_row.pack(fill="x", pady=(6, 0))
        self.output_entry = ctk.CTkEntry(
            out_row,
            placeholder_text="./compressed-model",
            height=40,
            font=ctk.CTkFont(size=13),
        )
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkButton(
            out_row,
            text="Browse",
            width=100,
            height=40,
            command=self._browse_output,
        ).pack(side="right")

        # --- Target mode ---
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", pady=16)
        ctk.CTkLabel(row, text="Compression Target", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")

        mode_row = ctk.CTkFrame(row, fg_color="transparent")
        mode_row.pack(fill="x", pady=(8, 0))
        self.target_mode = ctk.CTkSegmentedButton(
            mode_row,
            values=["Parameters", "Size (GB)"],
            font=ctk.CTkFont(size=13),
            command=self._on_target_mode_change,
        )
        self.target_mode.set("Parameters")
        self.target_mode.pack(side="left", padx=(0, 15))

        self.target_entry = ctk.CTkEntry(
            mode_row,
            placeholder_text="3400000000",
            width=260,
            height=40,
            font=ctk.CTkFont(size=13),
        )
        self.target_entry.pack(side="left")

        self.target_hint = ctk.CTkLabel(
            mode_row,
            text="target parameter count",
            font=ctk.CTkFont(size=12),
            text_color="gray60",
        )
        self.target_hint.pack(side="left", padx=(12, 0))

        # --- Advanced options ---
        adv = ctk.CTkFrame(scroll, corner_radius=12)
        adv.pack(fill="x", pady=20)

        ctk.CTkLabel(
            adv,
            text="Advanced Options",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(15, 12))

        grid = ctk.CTkFrame(adv, fg_color="transparent")
        grid.pack(fill="x", padx=18, pady=(0, 18))
        grid.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # LoRA r
        ctk.CTkLabel(grid, text="LoRA r", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.lora_r = ctk.CTkEntry(grid, placeholder_text="16", font=ctk.CTkFont(size=13))
        self.lora_r.insert(0, "16")
        self.lora_r.grid(row=1, column=0, sticky="ew", padx=8, pady=2)

        # LoRA alpha
        ctk.CTkLabel(grid, text="LoRA α", font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w", padx=8, pady=6)
        self.lora_alpha = ctk.CTkEntry(grid, placeholder_text="32", font=ctk.CTkFont(size=13))
        self.lora_alpha.insert(0, "32")
        self.lora_alpha.grid(row=1, column=1, sticky="ew", padx=8, pady=2)

        # Finetune steps
        ctk.CTkLabel(grid, text="Recovery Steps", font=ctk.CTkFont(size=12)).grid(row=0, column=2, sticky="w", padx=8, pady=6)
        self.steps = ctk.CTkEntry(grid, placeholder_text="500", font=ctk.CTkFont(size=13))
        self.steps.insert(0, "500")
        self.steps.grid(row=1, column=2, sticky="ew", padx=8, pady=2)

        # Max seq length
        ctk.CTkLabel(grid, text="Max Seq Length", font=ctk.CTkFont(size=12)).grid(row=0, column=3, sticky="w", padx=8, pady=6)
        self.max_seq = ctk.CTkEntry(grid, placeholder_text="512", font=ctk.CTkFont(size=13))
        self.max_seq.insert(0, "512")
        self.max_seq.grid(row=1, column=3, sticky="ew", padx=8, pady=2)

        # Dataset
        ctk.CTkLabel(grid, text="Recovery Dataset", font=ctk.CTkFont(size=12)).grid(row=2, column=0, sticky="w", padx=8, pady=14)
        self.dataset = ctk.CTkEntry(grid, placeholder_text="yahma/alpaca-cleaned", font=ctk.CTkFont(size=13))
        self.dataset.insert(0, "yahma/alpaca-cleaned")
        self.dataset.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=2)

        # Torch dtype
        ctk.CTkLabel(grid, text="Torch dtype", font=ctk.CTkFont(size=12)).grid(row=2, column=2, sticky="w", padx=8, pady=14)
        self.dtype = ctk.CTkOptionMenu(grid, values=["float16", "bfloat16", "float32"], font=ctk.CTkFont(size=13))
        self.dtype.set("float16")
        self.dtype.grid(row=3, column=2, sticky="ew", padx=8, pady=2)

        # Device
        ctk.CTkLabel(grid, text="Device Map", font=ctk.CTkFont(size=12)).grid(row=2, column=3, sticky="w", padx=8, pady=14)
        self.device = ctk.CTkOptionMenu(grid, values=["auto", "cuda", "cpu"], font=ctk.CTkFont(size=13))
        self.device.set("auto")
        self.device.grid(row=3, column=3, sticky="ew", padx=8, pady=2)

        # Export format
        ctk.CTkLabel(grid, text="Export Format", font=ctk.CTkFont(size=12)).grid(row=4, column=0, sticky="w", padx=8, pady=14)
        self.export_format = ctk.CTkOptionMenu(grid, values=["hf", "gguf", "onnx"], font=ctk.CTkFont(size=13))
        self.export_format.set("hf")
        self.export_format.grid(row=5, column=0, sticky="ew", padx=8, pady=2)

        # Selection method
        ctk.CTkLabel(grid, text="Neuron Selection", font=ctk.CTkFont(size=12)).grid(row=4, column=1, sticky="w", padx=8, pady=14)
        self.selection_method = ctk.CTkOptionMenu(grid, values=["greedy", "qubo"], font=ctk.CTkFont(size=13))
        self.selection_method.set("greedy")
        self.selection_method.grid(row=5, column=1, sticky="ew", padx=8, pady=2)

        # --- Buttons ---
        btn_row = ctk.CTkFrame(scroll, fg_color="transparent")
        btn_row.pack(fill="x", pady=(10, 30))
        self.start_btn = ctk.CTkButton(
            btn_row,
            text="Start Compression",
            width=200,
            height=48,
            corner_radius=10,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._start_compression,
        )
        self.start_btn.pack(side="left", padx=(0, 15))

        self.plan_btn = ctk.CTkButton(
            btn_row,
            text="Plan Only (Dry Run)",
            width=180,
            height=48,
            corner_radius=10,
            fg_color="#2c3e50",
            hover_color="#34495e",
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._start_dry_run,
        )
        self.plan_btn.pack(side="left", padx=(0, 15))

        self.validate_label = ctk.CTkLabel(
            btn_row,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="#e74c3c",
        )
        self.validate_label.pack(side="left")

    def _on_target_mode_change(self, mode: str):
        if mode == "Parameters":
            self.target_entry.configure(placeholder_text="3400000000")
            self.target_entry.delete(0, "end")
            self.target_hint.configure(text="target parameter count")
        else:
            self.target_entry.configure(placeholder_text="3.5")
            self.target_entry.delete(0, "end")
            self.target_hint.configure(text="target size in GB (fp16/bf16)")

    def _browse_model(self):
        path = filedialog.askdirectory(title="Select Model Folder")
        if path:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    # -----------------------------------------------------------------------
    # Progress view
    # -----------------------------------------------------------------------
    def _build_progress(self, parent: ctk.CTkFrame):
        header = ctk.CTkLabel(
            parent,
            text="Compression Progress",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        header.pack(anchor="w", padx=30, pady=(25, 15))

        # Overall progress
        self.overall_bar = ctk.CTkProgressBar(parent, height=18, corner_radius=8)
        self.overall_bar.pack(fill="x", padx=30, pady=(0, 20))
        self.overall_bar.set(0)

        # Stage cards
        self.stage_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.stage_frame.pack(fill="x", padx=30, pady=5)
        self.stage_frame.grid_columnconfigure(0, weight=1)
        self.stage_frame.grid_columnconfigure(1, weight=3)

        for i, name in enumerate(self.stages):
            num = ctk.CTkLabel(
                self.stage_frame,
                text=f"{i + 1}.",
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color="gray60",
                width=30,
            )
            num.grid(row=i, column=0, sticky="w", padx=(0, 10), pady=8)

            label = ctk.CTkLabel(
                self.stage_frame,
                text=name,
                font=ctk.CTkFont(size=14),
                text_color="gray70",
            )
            label.grid(row=i, column=1, sticky="w", pady=8)
            self.stage_labels.append(label)

            bar = ctk.CTkProgressBar(self.stage_frame, width=180, height=10, corner_radius=5)
            bar.grid(row=i, column=2, sticky="e", pady=8)
            bar.set(0)
            self.stage_bars.append(bar)

        # Log console
        log_frame = ctk.CTkFrame(parent, corner_radius=12)
        log_frame.pack(fill="both", expand=True, padx=30, pady=(25, 20))

        ctk.CTkLabel(
            log_frame,
            text="Execution Log",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=15, pady=(12, 8))

        self.log_box = ctk.CTkTextbox(
            log_frame,
            font=ctk.CTkFont(size=12, family="Consolas"),
            wrap="word",
            corner_radius=8,
        )
        self.log_box.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        self.log_box.configure(state="disabled")

        # Bottom controls
        btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=(0, 20))

        self.cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            width=130,
            height=40,
            fg_color="#c0392b",
            hover_color="#922b21",
            command=self._cancel_compression,
        )
        self.cancel_btn.pack(side="right", padx=(10, 0))
        self.cancel_btn.configure(state="disabled")

        self.back_btn = ctk.CTkButton(
            btn_frame,
            text="Back to Compress",
            width=160,
            height=40,
            command=lambda: self._show_view("compress"),
        )
        self.back_btn.pack(side="right")

    # -----------------------------------------------------------------------
    # About view
    # -----------------------------------------------------------------------
    def _build_about(self, parent: ctk.CTkFrame):
        ctk.CTkLabel(
            parent,
            text="About Qunart v2",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(anchor="w", padx=35, pady=(35, 15))

        text = (
            "Qunart v2 is a universal LLM compression framework.\n\n"
            "It takes any Hugging Face causal language model, profiles it, plans a "
            "structured width reduction, prunes the weights using a QUBO/greedy "
            "neuron selector, recovers quality with LoRA fine-tuning, and exports "
            "a valid smaller checkpoint.\n\n"
            "Supported model families: Llama, Qwen2, Mistral, Phi-3.\n\n"
            "This is a research scaffold. Always evaluate compressed models before deployment."
        )

        ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=14),
            text_color="gray80",
            wraplength=800,
            justify="left",
        ).pack(anchor="w", padx=35, pady=10)

    # -----------------------------------------------------------------------
    # Compression logic
    # -----------------------------------------------------------------------
    def _start_compression(self):
        if self.is_running:
            return

        self.validate_label.configure(text="")

        # Validate inputs
        model = self.model_entry.get().strip()
        output = self.output_entry.get().strip()
        target_text = self.target_entry.get().strip()
        mode = self.target_mode.get()

        if not model:
            self.validate_label.configure(text="Model path is required.")
            return
        if not output:
            self.validate_label.configure(text="Output directory is required.")
            return
        if not target_text:
            self.validate_label.configure(text="Target is required.")
            return

        try:
            if mode == "Parameters":
                target_params = int(target_text)
                target_size = None
            else:
                target_size = float(target_text)
                target_params = None
        except ValueError:
            self.validate_label.configure(text="Target must be a number.")
            return

        try:
            lora_r = int(self.lora_r.get().strip() or "16")
            lora_alpha = int(self.lora_alpha.get().strip() or "32")
            steps = int(self.steps.get().strip() or "500")
            max_seq = int(self.max_seq.get().strip() or "512")
        except ValueError:
            self.validate_label.configure(text="LoRA and step values must be integers.")
            return

        dataset = self.dataset.get().strip() or "yahma/alpaca-cleaned"
        dtype = self.dtype.get()
        device = self.device.get()
        export_fmt = self.export_format.get()
        sel_method = self.selection_method.get()

        target = CompressionTarget(
            target_params=target_params,
            target_size_gb=target_size,
            finetune_steps=steps,
            finetune_lr=2e-4,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            max_seq_length=max_seq,
            batch_size=1,
            gradient_accumulation_steps=4,
            recovery_dataset=dataset,
            torch_dtype=dtype,  # type: ignore
            device=device,
            selection_method=sel_method,
        )

        self._reset_progress()
        self._show_view("progress")
        self._set_running(True)
        self._append_log(f"[{datetime.now():%H:%M:%S}] Starting compression...\n")

        self.compression_thread = threading.Thread(
            target=self._compression_worker,
            args=(model, output, target, dataset, export_fmt),
            daemon=True,
        )
        self.compression_thread.start()

    def _start_dry_run(self):
        if self.is_running:
            return

        self.validate_label.configure(text="")
        model = self.model_entry.get().strip()
        target_text = self.target_entry.get().strip()
        mode = self.target_mode.get()

        if not model:
            self.validate_label.configure(text="Model path is required for dry run.")
            return

        target_params = None
        target_size = None
        if target_text:
            try:
                if mode == "Parameters":
                    target_params = int(target_text)
                else:
                    target_size = float(target_text)
            except ValueError:
                self.validate_label.configure(text="Target must be a number.")
                return

        dtype = self.dtype.get()
        target = CompressionTarget(
            target_params=target_params,
            target_size_gb=target_size,
            torch_dtype=dtype,  # type: ignore
        )

        self._reset_progress()
        self._show_view("progress")
        self._set_running(True)
        self._append_log(f"[{datetime.now():%H:%M:%S}] Planning compression (dry run)...\n")

        def _dry_worker():
            sys.stdout = QueueStdout(self.log_queue)
            try:
                pipeline = CompressionPipeline(target)
                pipeline.dry_run(model)
                self.log_queue.put("__DONE__")
            except Exception as exc:
                self.log_queue.put(f"__ERROR__{exc}")
            finally:
                sys.stdout = self.old_stdout

        self.compression_thread = threading.Thread(target=_dry_worker, daemon=True)
        self.compression_thread.start()

    def _compression_worker(self, model: str, output: str, target: CompressionTarget, dataset: str, export_format: str):
        """Run the pipeline in a background thread with captured stdout."""
        sys.stdout = QueueStdout(self.log_queue)
        try:
            pipeline = CompressionPipeline(target)
            pipeline.run(model, output, dataset_name=dataset, export_format=export_format)
            self.log_queue.put("__DONE__")
        except Exception as exc:
            self.log_queue.put(f"__ERROR__{exc}")
        finally:
            sys.stdout = self.old_stdout

    def _cancel_compression(self):
        """User-facing cancel. Python threads cannot be force-killed, so we warn."""
        if self.is_running and self.compression_thread and self.compression_thread.is_alive():
            messagebox.showinfo(
                "Cancel",
                "Compression is already running. The safest way to stop is to close the application."
            )

    def _set_running(self, running: bool):
        self.is_running = running
        if running:
            self.start_btn.configure(state="disabled", text="Running...")
            self.cancel_btn.configure(state="normal")
            self.status_label.configure(text="Running compression")
        else:
            self.start_btn.configure(state="normal", text="Start Compression")
            self.cancel_btn.configure(state="disabled")
            self.status_label.configure(text="Ready")

    def _reset_progress(self):
        self.overall_bar.set(0)
        for label in self.stage_labels:
            label.configure(text_color="gray70")
        for bar in self.stage_bars:
            bar.set(0)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _append_log(self, text: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _start_queue_polling(self):
        self._poll_queue()

    def _poll_queue(self):
        """Poll log queue from the main thread and update UI."""
        try:
            while True:
                item = self.log_queue.get_nowait()

                if item == "__DONE__":
                    self._set_running(False)
                    self.overall_bar.set(1.0)
                    self._update_stage(4, 1.0)
                    self._append_log(f"\n[{datetime.now():%H:%M:%S}] ✅ Compression finished successfully.\n")
                    messagebox.showinfo("Done", "Compression finished successfully.")
                    continue

                if isinstance(item, str) and item.startswith("__ERROR__"):
                    error = item.replace("__ERROR__", "")
                    self._set_running(False)
                    self._append_log(f"\n[{datetime.now():%H:%M:%S}] ❌ Error: {error}\n")
                    messagebox.showerror("Compression Error", str(error))
                    continue

                self._append_log(item)
                self._parse_progress(item)
        except queue.Empty:
            pass

        self.after(100, self._poll_queue)

    def _parse_progress(self, text: str):
        """Map pipeline stage markers to UI state."""
        # Match markers like [1/5], [2/5], etc.
        match = re.search(r"\[(\d)/5\]", text)
        if match:
            stage = int(match.group(1)) - 1
            if 0 <= stage < 5:
                # mark previous stages as done
                for i in range(stage):
                    self._update_stage(i, 1.0)
                    self.stage_labels[i].configure(text_color="#2ecc71")
                # pulse current stage (indeterminate progress)
                self._update_stage(stage, 0.5)
                self.stage_labels[stage].configure(text_color=("#1f6aa5", "#5dade2"))
                self.overall_bar.set((stage + 0.5) / 5.0)

        # Finished a stage when the next marker appears is handled above.
        # Final completion is handled by __DONE__.

    def _update_stage(self, index: int, value: float):
        if 0 <= index < len(self.stage_bars):
            self.stage_bars[index].set(value)

    # -----------------------------------------------------------------------
    # Safe exit
    # -----------------------------------------------------------------------
    def on_close(self):
        if self.is_running and self.compression_thread and self.compression_thread.is_alive():
            if not messagebox.askyesno(
                "Quit?",
                "Compression is still running. Closing now may leave a partial model. Are you sure?"
            ):
                return
        sys.stdout = self.old_stdout
        self.destroy()


def main():
    app = QunartApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
