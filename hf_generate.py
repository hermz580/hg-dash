#!/usr/bin/env python3
"""
Apache Typer + Hugging Face Generation Dashboard
A beautiful CLI interface for AI text generation with artistic flair
Enhanced with better architecture, error handling, and advanced features
"""

import typer
from typing import Optional, List, Dict, Any, Union
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.layout import Layout
from rich.align import Align
from rich.tree import Tree
from rich.status import Status
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from contextlib import contextmanager

# Hugging Face imports with better error handling
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
    import torch
    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    IMPORT_ERROR = str(e)

app = typer.Typer(
    name="hf-generate",
    help="🎨 Apache Typer + Hugging Face Generation Dashboard - Where Art Meets AI",
    rich_markup_mode="rich",
    no_args_is_help=True
)

console = Console()

# Enhanced ASCII Art Banner
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██╗  ██╗███████╗     ██████╗ ███████╗███╗   ██╗    ██████╗  █████╗ ███████╗    ║
║    ██║  ██║██╔════╝    ██╔════╝ ██╔════╝████╗  ██║    ██╔══██╗██╔══██╗██╔════╝    ║
║    ███████║█████╗      ██║  ███╗█████╗  ██╔██╗ ██║    ██║  ██║███████║███████╗    ║
║    ██╔══██║██╔══╝      ██║   ██║██╔══╝  ██║╚██╗██║    ██║  ██║██╔══██║╚════██║    ║
║    ██║  ██║██║         ╚██████╔╝███████╗██║ ╚████║    ██████╔╝██║  ██║███████║    ║
║    ╚═╝  ╚═╝╚═╝          ╚═════╝ ╚══════╝╚═╝  ╚═══╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝    ║
║                                                                              ║
║                   🎭 Artistic AI Generation Dashboard 🎨                    ║
║                        Enhanced Edition v2.0                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

class GenerationStyle(str, Enum):
    """Enhanced generation styles with specific configurations"""
    CREATIVE = "creative"
    PRECISE = "precise"
    EXPERIMENTAL = "experimental"
    BALANCED = "balanced"
    STORYTELLING = "storytelling"
    TECHNICAL = "technical"

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_length: int = 150
    min_length: int = 10
    do_sample: bool = True

@dataclass
class GenerationResult:
    """Result of text generation with metadata"""
    prompt: str
    generated_text: str
    timestamp: str
    model_name: str
    config: GenerationConfig
    generation_time: float
    token_count: int

class ConfigManager:
    """Manage application configuration and settings"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".hf-generate"
        self.config_file = self.config_dir / "config.json"
        self.history_file = self.config_dir / "history.json"
        self.config_dir.mkdir(exist_ok=True)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "default_model": "microsoft/DialoGPT-medium",
            "default_style": "creative",
            "auto_save_history": True,
            "max_history_entries": 100,
            "theme": "monokai"
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return {**default_config, **json.load(f)}
            except Exception as e:
                console.print(f"⚠️ [yellow]Config load error: {e}. Using defaults.[/yellow]")
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            console.print(f"⚠️ [yellow]Config save error: {e}[/yellow]")

class ModelManager:
    """Enhanced model management with caching and validation"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.models = {}
        self.current_model = None
        self.device = self._detect_device()
        self.model_cache = {}
        
    def _detect_device(self) -> str:
        """Intelligently detect the best available device"""
        if not HF_AVAILABLE:
            return "cpu"
            
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"🚀 [green]CUDA detected: {gpu_count} GPU(s) - {gpu_name}[/green]")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            console.print("🍎 [green]Apple Silicon MPS detected[/green]")
            return "mps"
        else:
            console.print("💻 [yellow]Using CPU (consider GPU for better performance)[/yellow]")
            return "cpu"
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if model exists and is accessible"""
        try:
            config = AutoConfig.from_pretrained(model_name)
            return True
        except Exception:
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed model information"""
        try:
            config = AutoConfig.from_pretrained(model_name)
            return {
                "model_type": getattr(config, 'model_type', 'unknown'),
                "vocab_size": getattr(config, 'vocab_size', 'unknown'),
                "hidden_size": getattr(config, 'hidden_size', 'unknown'),
                "num_layers": getattr(config, 'num_hidden_layers', 'unknown')
            }
        except Exception:
            return {"error": "Could not fetch model info"}
    
    def load_model(self, model_name: str, task: str = "text-generation", force_reload: bool = False):
        """Load model with enhanced progress tracking and error handling"""
        if not HF_AVAILABLE:
            console.print(f"💥 [bold red]Transformers not available: {IMPORT_ERROR}[/bold red]")
            return False
        
        if model_name in self.models and not force_reload:
            self.current_model = model_name
            console.print(f"✅ [green]Model {model_name} already loaded![/green]")
            return True
        
        # Validate model first
        console.print(f"🔍 [blue]Validating model: {model_name}[/blue]")
        if not self.validate_model(model_name):
            console.print(f"❌ [red]Model {model_name} not found or inaccessible[/red]")
            return False
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task_id = progress.add_task(f"🧠 Loading {model_name}", total=100)
            
            try:
                # Stage 1: Initialize tokenizer
                progress.update(task_id, description="📝 Loading tokenizer...", advance=20)
                time.sleep(0.3)
                
                # Stage 2: Load model
                progress.update(task_id, description="🧠 Loading model...", advance=40)
                
                device_map = None
                if self.device == "cuda" and torch.cuda.device_count() > 1:
                    device_map = "auto"
                
                model = pipeline(
                    task,
                    model=model_name,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=device_map
                )
                
                progress.update(task_id, description="⚡ Optimizing...", advance=30)
                time.sleep(0.2)
                
                # Store model info
                model_info = self.get_model_info(model_name)
                
                progress.update(task_id, description="✅ Complete!", advance=10)
                
                self.models[model_name] = {
                    "pipeline": model,
                    "info": model_info,
                    "loaded_at": datetime.now().isoformat()
                }
                self.current_model = model_name
                
                console.print(f"✨ [bold green]Model {model_name} loaded successfully on {self.device}![/bold green]")
                self._display_model_info(model_name)
                return True
                
            except Exception as e:
                console.print(f"💥 [bold red]Failed to load {model_name}: {str(e)}[/bold red]")
                return False
    
    def _display_model_info(self, model_name: str):
        """Display model information in a nice format"""
        if model_name in self.models:
            info = self.models[model_name]["info"]
            table = Table(title=f"📊 Model Info: {model_name}", show_header=False)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in info.items():
                if key != "error":
                    table.add_row(key.replace('_', ' ').title(), str(value))
            
            console.print(table)
    
    def unload_model(self, model_name: str = None):
        """Unload a specific model or current model"""
        target = model_name or self.current_model
        if target and target in self.models:
            del self.models[target]
            if target == self.current_model:
                self.current_model = None
            console.print(f"🗑️ [yellow]Model {target} unloaded[/yellow]")
            return True
        return False

class GenerationStudio:
    """Enhanced text generation with advanced features"""
    
    def __init__(self, model_manager: ModelManager, config_manager: ConfigManager):
        self.model_manager = model_manager
        self.config_manager = config_manager
        self.generation_history: List[GenerationResult] = []
        self.style_configs = {
            GenerationStyle.CREATIVE: GenerationConfig(temperature=0.8, top_p=0.95, repetition_penalty=1.05),
            GenerationStyle.PRECISE: GenerationConfig(temperature=0.3, top_p=0.7, repetition_penalty=1.2),
            GenerationStyle.EXPERIMENTAL: GenerationConfig(temperature=1.0, top_p=0.98, top_k=100),
            GenerationStyle.BALANCED: GenerationConfig(temperature=0.7, top_p=0.9, repetition_penalty=1.1),
            GenerationStyle.STORYTELLING: GenerationConfig(temperature=0.8, top_p=0.9, max_length=300, repetition_penalty=1.05),
            GenerationStyle.TECHNICAL: GenerationConfig(temperature=0.4, top_p=0.8, repetition_penalty=1.3)
        }
        self.load_history()
    
    def load_history(self):
        """Load generation history from file"""
        if self.config_manager.history_file.exists():
            try:
                with open(self.config_manager.history_file, 'r') as f:
                    history_data = json.load(f)
                    # Handle old format gracefully
                    if history_data and isinstance(history_data[0], dict):
                        # Convert dict config to GenerationConfig if needed
                        for item in history_data:
                            if 'config' in item and isinstance(item['config'], dict):
                                item['config'] = GenerationConfig(**item['config'])
                    self.generation_history = [
                        GenerationResult(**item) if isinstance(item, dict) else item 
                        for item in history_data
                    ]
            except Exception as e:
                console.print(f"⚠️ [yellow]History load error: {e}[/yellow]")
    
    def save_history(self):
        """Save generation history to file"""
        if self.config_manager.config["auto_save_history"]:
            try:
                # Keep only the most recent entries
                max_entries = self.config_manager.config["max_history_entries"]
                recent_history = self.generation_history[-max_entries:]
                
                # Convert to dict for JSON serialization
                history_data = []
                for item in recent_history:
                    item_dict = asdict(item) if hasattr(item, '__dict__') else item
                    # Ensure config is a dict
                    if 'config' in item_dict and hasattr(item_dict['config'], '__dict__'):
                        item_dict['config'] = asdict(item_dict['config'])
                    history_data.append(item_dict)
                
                with open(self.config_manager.history_file, 'w') as f:
                    json.dump(history_data, f, indent=2)
            except Exception as e:
                console.print(f"⚠️ [yellow]History save error: {e}[/yellow]")
    
    def generate_text(
        self,
        prompt: str,
        style: GenerationStyle = GenerationStyle.CREATIVE,
        custom_config: Optional[GenerationConfig] = None
    ) -> Optional[GenerationResult]:
        """Generate text with comprehensive error handling and timing"""
        
        if not self.model_manager.current_model:
            console.print("💔 [bold red]No model loaded! Use 'load-model' command first.[/bold red]")
            return None
        
        model_data = self.model_manager.models[self.model_manager.current_model]
        model = model_data["pipeline"]
        
        # Use custom config or style-based config
        config = custom_config or self.style_configs[style]
        
        start_time = time.time()
        
        with Status(f"🎨 [bold magenta]Crafting your {style.value} creation...", console=console):
            try:
                result = model(
                    prompt,
                    max_length=config.max_length,
                    min_length=config.min_length,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    repetition_penalty=config.repetition_penalty,
                    do_sample=config.do_sample,
                    pad_token_id=model.tokenizer.eos_token_id if hasattr(model, 'tokenizer') else None
                )
                
                generation_time = time.time() - start_time
                generated_text = result[0]['generated_text']
                
                # Create result object
                generation_result = GenerationResult(
                    prompt=prompt,
                    generated_text=generated_text,
                    timestamp=datetime.now().isoformat(),
                    model_name=self.model_manager.current_model,
                    config=config,
                    generation_time=generation_time,
                    token_count=len(generated_text.split())
                )
                
                # Store in history
                self.generation_history.append(generation_result)
                self.save_history()
                
                return generation_result
                
            except Exception as e:
                console.print(f"💥 [bold red]Generation failed: {str(e)}[/bold red]")
                return None
    
    def batch_generate(self, prompts: List[str], style: GenerationStyle = GenerationStyle.CREATIVE) -> List[GenerationResult]:
        """Generate text for multiple prompts"""
        results = []
        
        with Progress(console=console) as progress:
            task = progress.add_task("🎭 Batch Generation", total=len(prompts))
            
            for prompt in prompts:
                result = self.generate_text(prompt, style)
                if result:
                    results.append(result)
                progress.advance(task)
        
        return results

# Initialize global instances
config_manager = ConfigManager()
model_manager = ModelManager(config_manager)
studio = GenerationStudio(model_manager, config_manager)

@app.command()
def splash():
    """🎭 Display the artistic banner"""
    console.print(BANNER, style="bold cyan")
    console.print("\n[italic]Welcome to the enhanced intersection of Apache Typer and Hugging Face![/italic]\n")
    
    # System info
    info_table = Table(show_header=False, box=None)
    info_table.add_column("", style="cyan")
    info_table.add_column("", style="white")
    
    info_table.add_row("🖥️  Device", model_manager.device.upper())
    info_table.add_row("🧠 HF Available", "✅ Yes" if HF_AVAILABLE else "❌ No")
    info_table.add_row("📁 Config Dir", str(config_manager.config_dir))
    
    console.print(Panel(info_table, title="System Info", border_style="blue"))

@app.command()
def load_model(
    model_name: str = typer.Argument(
        None,
        help="Model name from Hugging Face Hub"
    ),
    task: str = typer.Option(
        "text-generation",
        help="Task type (text-generation, text2text-generation, etc.)"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force reload if model already loaded"
    )
):
    """🧠 Load a Hugging Face model with enhanced validation"""
    
    if not model_name:
        model_name = config_manager.config["default_model"]
        console.print(f"📋 [blue]Using default model: {model_name}[/blue]")
    
    console.print(f"\n🎨 [bold]Loading the creative engine: {model_name}[/bold]\n")
    
    success = model_manager.load_model(model_name, task, force)
    if success:
        config_manager.config["default_model"] = model_name
        config_manager.save_config()

@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Your creative prompt"),
    max_length: int = typer.Option(150, help="Maximum generation length"),
    temperature: float = typer.Option(0.7, help="Creativity level (0.1-2.0)"),
    style: GenerationStyle = typer.Option(GenerationStyle.CREATIVE, help="Generation style"),
    save_output: bool = typer.Option(False, "--save", help="Save output to file"),
    model: str = typer.Option(None, "--model", help="Model to use (auto-loads if not specified)")
):
    """✨ Generate text with enhanced controls"""
    
    console.print(f"\n🎪 [bold]Generating with style: {style.value}[/bold]")
    console.print(Panel(prompt, title="📝 Your Prompt", border_style="blue"))
    
    # Auto-load model if not already loaded
    if not model_manager.current_model:
        model_to_use = model or config_manager.config["default_model"]
        console.print(f"🧠 [blue]Auto-loading model: {model_to_use}[/blue]")
        success = model_manager.load_model(model_to_use)
        if not success:
            console.print("💔 [bold red]Failed to load model![/bold red]")
            return
    elif model and model != model_manager.current_model:
        # User specified a different model
        console.print(f"🔄 [blue]Loading requested model: {model}[/blue]")
        success = model_manager.load_model(model)
        if not success:
            console.print("💔 [bold red]Failed to load requested model![/bold red]")
            return
    
    # Create custom config if parameters provided
    custom_config = GenerationConfig(
        max_length=max_length,
        temperature=temperature
    )
    
    result = studio.generate_text(prompt, style, custom_config)
    
    if result:
        # Beautiful output formatting
        console.print("\n" + "="*80)
        
        # Generation metadata
        metadata = f"⏱️ Generated in {result.generation_time:.2f}s | 📊 {result.token_count} tokens | 🎭 {style.value}"
        console.print(f"[dim]{metadata}[/dim]")
        
        console.print(Panel(
            Syntax(result.generated_text, "text", theme=config_manager.config["theme"], line_numbers=False),
            title="🎨 Generated Masterpiece",
            border_style="green"
        ))
        console.print("="*80 + "\n")
        
        if save_output:
            filename = f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Generated Text:\n{result.generated_text}\n\n")
                f.write(f"Metadata: {metadata}\n")
            console.print(f"💾 [green]Output saved to {filename}[/green]")

@app.command()
def batch(
    file: str = typer.Argument(..., help="File containing prompts (one per line)"),
    style: GenerationStyle = typer.Option(GenerationStyle.CREATIVE, help="Generation style"),
    output_dir: str = typer.Option("batch_output", help="Output directory")
):
    """🎭 Batch generate from file of prompts"""
    
    try:
        with open(file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        console.print(f"❌ [red]File not found: {file}[/red]")
        return
    
    console.print(f"📚 [blue]Processing {len(prompts)} prompts...[/blue]")
    
    results = studio.batch_generate(prompts, style)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i, result in enumerate(results):
        filename = f"{output_dir}/generation_{i+1:03d}_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(f"Prompt: {result.prompt}\n\n")
            f.write(f"Generated Text:\n{result.generated_text}\n")
    
    console.print(f"✅ [green]Batch complete! {len(results)} files saved to {output_dir}[/green]")

@app.command()
def interactive():
    """🎮 Enhanced interactive generation mode"""
    console.print(BANNER, style="bold cyan")
    console.print("\n🎭 [bold]Welcome to Interactive Generation Studio v2.0![/bold]\n")
    
    if not model_manager.current_model:
        if Confirm.ask("No model loaded. Would you like to load the default model?"):
            model_manager.load_model(config_manager.config["default_model"])
        else:
            return
    
    console.print("\n💡 [italic]Commands: 'quit', 'history', 'models', 'config', 'clear'[/italic]\n")
    
    while True:
        try:
            prompt = Prompt.ask("\n🎨 [bold cyan]Enter your creative prompt[/bold cyan]")
            
            if prompt.lower() == 'quit':
                console.print("👋 [bold]Thank you for creating with us![/bold]")
                break
            elif prompt.lower() == 'history':
                show_history()
                continue
            elif prompt.lower() == 'models':
                show_models()
                continue
            elif prompt.lower() == 'config':
                show_config()
                continue
            elif prompt.lower() == 'clear':
                console.clear()
                continue
            
            # Advanced parameter selection
            if Confirm.ask("🔧 Customize generation parameters?", default=False):
                style = Prompt.ask(
                    "🎭 Choose style",
                    choices=[s.value for s in GenerationStyle],
                    default=config_manager.config["default_style"]
                )
                max_length = IntPrompt.ask("📏 Max length", default=150)
                temperature = FloatPrompt.ask("🌡️ Temperature", default=0.7)
                
                custom_config = GenerationConfig(
                    max_length=max_length,
                    temperature=temperature
                )
                result = studio.generate_text(prompt, GenerationStyle(style), custom_config)
            else:
                style = GenerationStyle(config_manager.config["default_style"])
                result = studio.generate_text(prompt, style)
            
            if result:
                console.print(Panel(
                    f"{result.generated_text}\n\n[dim]⏱️ {result.generation_time:.2f}s | 📊 {result.token_count} tokens[/dim]",
                    title="✨ Your Creation",
                    border_style="magenta"
                ))
                
        except KeyboardInterrupt:
            console.print("\n👋 [bold]Goodbye![/bold]")
            break

@app.command()
def history(
    limit: int = typer.Option(10, help="Number of recent entries to show"),
    export: bool = typer.Option(False, "--export", help="Export history to file")
):
    """📚 Show enhanced generation history"""
    show_history(limit, export)

def show_history(limit: int = 10, export: bool = False):
    """Display generation history with enhanced formatting"""
    if not studio.generation_history:
        console.print("📭 [yellow]No generations yet! Start creating![/yellow]")
        return
    
    if export:
        filename = f"history_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        history_data = []
        for item in studio.generation_history:
            item_dict = asdict(item) if hasattr(item, '__dict__') else item
            if 'config' in item_dict and hasattr(item_dict['config'], '__dict__'):
                item_dict['config'] = asdict(item_dict['config'])
            history_data.append(item_dict)
        
        with open(filename, 'w') as f:
            json.dump(history_data, f, indent=2)
        console.print(f"💾 [green]History exported to {filename}[/green]")
        return
    
    table = Table(title="🎨 Generation History", show_header=True, header_style="bold magenta")
    table.add_column("🕐 Time", style="dim", width=8)
    table.add_column("📝 Prompt", style="cyan", width=30)
    table.add_column("🎭 Style", style="green", width=12)
    table.add_column("⏱️ Time", style="yellow", width=6)
    table.add_column("📊 Tokens", style="blue", width=8)
    table.add_column("🎨 Preview", style="white", width=40)
    
    recent_entries = studio.generation_history[-limit:]
    for entry in recent_entries:
        timestamp = datetime.fromisoformat(entry.timestamp).strftime("%H:%M:%S")
        prompt_preview = (entry.prompt[:27] + "...") if len(entry.prompt) > 30 else entry.prompt
        text_preview = (entry.generated_text[:37] + "...") if len(entry.generated_text) > 40 else entry.generated_text
        
        # Get style from config
        style_name = "unknown"
        if hasattr(entry.config, 'temperature'):
            # Try to match style based on temperature
            if entry.config.temperature == 0.8:
                style_name = "creative"
            elif entry.config.temperature == 0.3:
                style_name = "precise"
            elif entry.config.temperature == 1.0:
                style_name = "experimental"
            elif entry.config.temperature == 0.7:
                style_name = "balanced"
            elif entry.config.temperature == 0.4:
                style_name = "technical"
        
        table.add_row(
            timestamp,
            prompt_preview,
            style_name,
            f"{entry.generation_time:.1f}s",
            str(entry.token_count),
            text_preview
        )
    
    console.print(table)

def show_models():
    """Enhanced model display"""
    console.print("\n🎭 [bold]Model Gallery[/bold]\n")
    
    # Currently loaded models
    if model_manager.models:
        loaded_table = Table(title="🟢 Loaded Models", show_header=True, header_style="bold green")
        loaded_table.add_column("Model", style="cyan")
        loaded_table.add_column("Type", style="yellow")
        loaded_table.add_column("Loaded At", style="dim")
        loaded_table.add_column("Current", style="green")
        
        for name, data in model_manager.models.items():
            is_current = "✅" if name == model_manager.current_model else ""
            loaded_at = datetime.fromisoformat(data["loaded_at"]).strftime("%H:%M:%S")
            model_type = data["info"].get("model_type", "unknown")
            
            loaded_table.add_row(name, model_type, loaded_at, is_current)
        
        console.print(loaded_table)
    
    # Recommended models
    recommended_table = Table(title="🌟 Recommended Models", show_header=True, header_style="bold blue")
    recommended_table.add_column("Model Name", style="cyan")
    recommended_table.add_column("Best For", style="green")
    recommended_table.add_column("Size", style="yellow")
    recommended_table.add_column("Speed", style="magenta")
    
    recommended = [
        ("microsoft/DialoGPT-medium", "Conversational AI", "Medium", "Fast"),
        ("gpt2", "General text generation", "Small", "Very Fast"),
        ("distilgpt2", "Quick generation", "Small", "Very Fast"),
        ("microsoft/DialoGPT-large", "High-quality dialogue", "Large", "Slow"),
        ("EleutherAI/gpt-neo-125M", "Creative writing", "Small", "Fast"),
        ("facebook/blenderbot-400M-distill", "Chatbot", "Medium", "Fast")
    ]
    
    for model, use_case, size, speed in recommended:
        recommended_table.add_row(model, use_case, size, speed)
    
    console.print(recommended_table)

def show_config():
    """Display current configuration"""
    config_table = Table(title="⚙️ Configuration", show_header=False)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    for key, value in config_manager.config.items():
        config_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(config_table)

@app.command()
def models():
    """🧠 Show enhanced model information"""
    show_models()

@app.command()
def config(
    key: str = typer.Argument(None, help="Configuration key to modify"),
    value: str = typer.Argument(None, help="New value")
):
    """⚙️ View or modify configuration"""
    if not key:
        show_config()
        return
    
    if key not in config_manager.config:
        console.print(f"❌ [red]Unknown configuration key: {key}[/red]")
        return
    
    if value:
        # Type conversion based on current value type
        current_value = config_manager.config[key]
        if isinstance(current_value, bool):
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(current_value, int):
            value = int(value)
        elif isinstance(current_value, float):
            value = float(value)
        
        config_manager.config[key] = value
        config_manager.save_config()
        console.print(f"✅ [green]Set {key} = {value}[/green]")
    else:
        console.print(f"{key}: {config_manager.config[key]}")

@app.command()
def unload(
    model_name: str = typer.Argument(None, help="Model to unload (current if not specified)")
):
    """🗑️ Unload a model to free memory"""
    success = model_manager.unload_model(model_name)
    if not success:
        console.print("❌ [red]No model to unload[/red]")

@app.command()
def stats():
    """📊 Show enhanced session statistics"""
    if not studio.generation_history:
        console.print("📭 [yellow]No generations yet![/yellow]")
        return
    
    total_generations = len(studio.generation_history)
    total_tokens = sum(entry.token_count for entry in studio.generation_history)
    avg_time = sum(entry.generation_time for entry in studio.generation_history) / total_generations
    
    # Style usage (approximate based on temperature)
    style_usage = {}
    for entry in studio.generation_history:
        if hasattr(entry.config, 'temperature'):
            temp = entry.config.temperature
            if temp == 0.8:
                style = "creative"
            elif temp == 0.3:
                style = "precise"
            elif temp == 1.0:
                style = "experimental"
            elif temp == 0.7:
                style = "balanced"
            elif temp == 0.4:
                style = "technical"
            else:
                style = "custom"
            style_usage[style] = style_usage.get(style, 0) + 1
    
    # Recent activity (last hour)
    recent_count = sum(1 for entry in studio.generation_history 
                      if (datetime.now() - datetime.fromisoformat(entry.timestamp)).seconds < 3600)
    
    stats_panel = Panel(
        f"🎨 Total Generations: {total_generations}\n"
        f"📝 Total Tokens: {total_tokens:,}\n"
        f"⏱️ Average Time: {avg_time:.2f}s\n"
        f"🔥 Recent Activity: {recent_count} (last hour)\n"
        f"🧠 Current Model: {model_manager.current_model or 'None'}\n"
        f"🎭 Popular Styles: {', '.join(sorted(style_usage.keys()))}",
        title="📊 Session Statistics",
        border_style="blue"
    )
    
    console.print(stats_panel)

@app.command()
def export_session(
    filename: str = typer.Option(
        None,
        help="Export filename (auto-generated if not provided)"
    ),
    format: str = typer.Option("json", help="Export format (json, csv, txt)")
):
    """💾 Export session with multiple formats"""
    if not filename:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hf_session_{timestamp}.{format}"
    
    try:
        if format == "json":
            session_data = {
                "timestamp": datetime.now().isoformat(),
                "current_model": model_manager.current_model,
                "device": model_manager.device,
                "config": config_manager.config,
                "generation_history": [asdict(entry) for entry in studio.generation_history],
                "model_info": {name: data["info"] for name, data in model_manager.models.items()}
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
                
        elif format == "csv":
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Prompt', 'Generated Text', 'Model', 'Generation Time', 'Token Count'])
                for entry in studio.generation_history:
                    writer.writerow([
                        entry.timestamp, entry.prompt, entry.generated_text,
                        entry.model_name, entry.generation_time, entry.token_count
                    ])
                    
        elif format == "txt":
            with open(filename, 'w') as f:
                f.write(f"HF Generation Session Export\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
                f.write(f"Current Model: {model_manager.current_model}\n\n")
                
                for i, entry in enumerate(studio.generation_history, 1):
                    f.write(f"=== Generation {i} ===\n")
                    f.write(f"Time: {entry.timestamp}\n")
                    f.write(f"Prompt: {entry.prompt}\n")
                    f.write(f"Generated: {entry.generated_text}\n")
                    f.write(f"Stats: {entry.generation_time:.2f}s, {entry.token_count} tokens\n\n")
        
        console.print(f"💾 [bold green]Session exported to {filename}[/bold green]")
        
    except Exception as e:
        console.print(f"💥 [bold red]Export failed: {str(e)}[/bold red]")

@app.command()
def benchmark(
    prompt: str = typer.Option("The future of AI is", help="Benchmark prompt"),
    iterations: int = typer.Option(5, help="Number of iterations"),
    styles: List[GenerationStyle] = typer.Option([GenerationStyle.CREATIVE], help="Styles to benchmark")
):
    """🏃 Benchmark generation performance"""
    if not model_manager.current_model:
        console.print("💔 [bold red]No model loaded![/bold red]")
        return
    
    console.print(f"🏃 [bold]Benchmarking {model_manager.current_model}[/bold]")
    console.print(f"📝 Prompt: {prompt}")
    console.print(f"🔄 Iterations: {iterations}")
    
    results = {}
    
    for style in styles:
        console.print(f"\n🎭 Testing style: {style.value}")
        times = []
        
        with Progress(console=console) as progress:
            task = progress.add_task(f"Benchmarking {style.value}", total=iterations)
            
            for i in range(iterations):
                result = studio.generate_text(f"{prompt} (iteration {i+1})", style)
                if result:
                    times.append(result.generation_time)
                progress.advance(task)
        
        if times:
            results[style.value] = {
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "total_time": sum(times)
            }
    
    # Display results
    bench_table = Table(title="🏆 Benchmark Results", show_header=True, header_style="bold yellow")
    bench_table.add_column("Style", style="cyan")
    bench_table.add_column("Avg Time", style="green")
    bench_table.add_column("Min Time", style="blue")
    bench_table.add_column("Max Time", style="red")
    bench_table.add_column("Total Time", style="magenta")
    
    for style, stats in results.items():
        bench_table.add_row(
            style,
            f"{stats['avg_time']:.2f}s",
            f"{stats['min_time']:.2f}s",
            f"{stats['max_time']:.2f}s",
            f"{stats['total_time']:.2f}s"
        )
    
    console.print(bench_table)

if __name__ == "__main__":
    # Check dependencies
    if not HF_AVAILABLE:
        console.print(f"⚠️ [bold yellow]Warning: Transformers not available: {IMPORT_ERROR}[/bold yellow]")
        console.print("Install with: pip install transformers torch\n")
    
    # Show splash screen on startup
    console.print(BANNER, style="bold cyan")
    console.print("[italic]Enhanced Edition - Use --help for available commands[/italic]\n")
    app()