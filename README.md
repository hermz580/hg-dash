# 🎨 HF Generate - Enhanced CLI Tool

## Apache Typer + Hugging Face Generation Dashboard
*A beautiful CLI interface for AI text generation with artistic flair*

### ✨ Features

- 🧠 **Smart Model Management**: Load, validate, and manage multiple Hugging Face models
- 🎭 **Multiple Generation Styles**: Creative, Precise, Experimental, Balanced, Storytelling, Technical
- 🖥️ **Beautiful Console Interface**: Rich formatting, progress bars, and ASCII art
- 📊 **Performance Tracking**: Generation timing, token counting, and statistics
- 💾 **Persistent History**: Auto-save generation history with export options
- ⚙️ **Configuration Management**: Customizable settings and themes
- 🔄 **Batch Processing**: Generate from files with multiple prompts
- 📈 **Benchmarking**: Performance testing across different styles
- 🎮 **Interactive Mode**: Real-time generation with live feedback

### 🚀 Quick Start

#### Running the Web Application (Frontend and Backend)
To run the web application, which includes a frontend and a backend server, you can use the provided development script. This will start both servers and handle dependencies for you.
```bash
./start-dev.sh
```
This will:
- Start the backend server on `http://localhost:8005`.
- Start the frontend development server on `http://localhost:3000` (usually).
- Open the application in your default web browser.

---

#### 1. Load a Model
```bash
python hf_generate.py load-model gpt2
```

#### 2. Generate Text
```bash
python hf_generate.py generate "The future of AI is" --max-length 100 --style creative
```

#### 3. Interactive Mode (Recommended)
```bash
python hf_generate.py interactive
```

#### 4. Batch Generation
```bash
python hf_generate.py batch prompts.txt --style storytelling
```

### 📋 Available Commands

| Command | Description |
|---------|-------------|
| `splash` | 🎭 Display the artistic banner |
| `load-model` | 🧠 Load a Hugging Face model |
| `generate` | ✨ Generate text with controls |
| `interactive` | 🎮 Interactive generation mode |
| `batch` | 🎭 Batch generate from file |
| `history` | 📚 Show generation history |
| `models` | 🧠 Show model information |
| `config` | ⚙️ View/modify configuration |
| `stats` | 📊 Show session statistics |
| `benchmark` | 🏃 Performance benchmarking |
| `export-session` | 💾 Export data |

### 🎭 Generation Styles

- **Creative** (0.8 temp): High creativity, diverse outputs
- **Precise** (0.3 temp): Focused, deterministic results
- **Experimental** (1.0 temp): Maximum randomness and variety
- **Balanced** (0.7 temp): Good mix of creativity and coherence
- **Storytelling** (0.8 temp): Optimized for narrative content
- **Technical** (0.4 temp): Factual, structured outputs

### 🖥️ System Requirements

- Python 3.8+
- PyTorch
- Transformers
- Rich
- Typer
- CUDA (optional, for GPU acceleration)

### 📊 Example Usage

#### Generate with Custom Parameters
```bash
python hf_generate.py generate "Write a poem about" \
  --max-length 200 \
  --temperature 0.9 \
  --style creative \
  --save
```

#### Benchmark Performance
```bash
python hf_generate.py benchmark \
  --prompt "The future of technology" \
  --iterations 10 \
  --styles creative precise experimental
```

#### Export Session Data
```bash
python hf_generate.py export-session \
  --filename my_session \
  --format json
```

### 🎯 Pro Tips

1. **Start with Interactive Mode**: Best way to explore the tool
2. **Use Small Models First**: gpt2, distilgpt2 for quick testing  
3. **GPU Acceleration**: Install CUDA for faster generation
4. **Batch Processing**: Use for large-scale content generation
5. **History Export**: Save important generations for later

### 🔧 Configuration

The tool creates a `~/.hf-generate/` directory with:
- `config.json`: User preferences and settings
- `history.json`: Generation history (auto-managed)

### 📈 Performance

- **CPU Mode**: Works on any system, slower generation
- **GPU Mode**: 5-10x faster with CUDA-enabled GPU
- **Model Size**: Smaller models = faster generation
- **Memory**: Larger models require more RAM/VRAM

---

## 🎨 Your Personal AI Generation Studio is Ready!

Start with: `python hf_generate.py interactive`

Happy generating! 🚀✨
