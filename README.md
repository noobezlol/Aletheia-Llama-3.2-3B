# Aletheia-Llama-3.2-3B

<div align="center">

![Uncensored](https://img.shields.io/badge/Uncensored-red?style=for-the-badge)
![Size](https://img.shields.io/badge/Size-3B-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-Llama%203.2%20Community-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-orange?style=for-the-badge)

# Uncensored Language Models

**High-performance AI models without content restrictions**

[Quick Start](#quick-start) • [Installation](#installation) • [Usage](#usage) • [Models](#models) • [API](#api-reference)

</div>

---

## Overview

This repository contains uncensored language models based on the Llama 3.2 architecture. These models are designed for research purposes and provide unfiltered responses while maintaining high-quality language generation.

> **Important**: These models are intended for research and development only. Users are responsible for ensuring compliance with applicable laws and regulations.

## LEGAL & ETHICAL DISCLAIMER

**READ THIS BEFORE DOWNLOADING OR USING**

### CRITICAL WARNING

This model is a **Proof-of-Concept (PoC)** designed **exclusively** for:
- AI Safety Research
- Red Teaming and Vulnerability Assessment  
- Alignment Research and Testing
- Academic and Educational Purposes

### TERMS OF USE

1. **Research Only**: This model is intended **solely** for authorized research in controlled environments. Any other use is strictly prohibited.

2. **No Liability**: The creators, contributors, and maintainers assume **ABSOLUTELY NO RESPONSIBILITY OR LIABILITY** for any:
   - Misuse of this software
   - Damage caused by deployment
   - Illegal activities conducted using this model
   - Violation of laws or regulations
   - **Users assume ALL responsibility for their actions**

3. **Prohibited Uses**: Use of this model for any of the following is **STRICTLY FORBIDDEN**:
   - Generation of malicious code or malware
   - Creation of biological or chemical weapon instructions
   - Harassment, threats, or targeted abuse
   - Fraud, scams, or deceptive practices
   - Any illegal content generation
   - **Violation of these terms constitutes a breach of the Llama 3.2 Community License**

4. **Authorized Environments**: This model should **ONLY** be used in:
   - Isolated research environments (sandboxes)
   - Authorized testing facilities
   - Academic institutions with proper oversight
   - Corporate security research with management approval

5. **Legal Compliance**: Users must ensure compliance with:
   - Local, state, and federal laws
   - The Llama 3.2 Community License terms
   - Institutional policies and guidelines
   - International regulations where applicable

### LEGAL ACKNOWLEDGMENT

**By downloading, installing, or using this model, you explicitly agree to:**

- Use this software **ONLY** for legitimate research purposes
- Assume **FULL LIABILITY** for any consequences of use
- Indemnify and hold harmless all creators and contributors
- Comply with all applicable laws and regulations
- Accept that this software is provided "AS IS" without warranties

**VIOLATION OF THESE TERMS MAY RESULT IN LEGAL ACTION AND IMMEDIATE REVOCATION OF ACCESS.**

## Features

| Feature | Description |
|---------|-------------|
| **High Performance** | Optimized for both speed and quality |
| **Easy Integration** | Simple API compatible with popular frameworks |
| **Multiple Formats** | Available in various model formats |
| **Docker Support** | Containerized deployment options |
| **Benchmark Results** | Comprehensive performance metrics |
| **Research Focused** | Designed for AI safety research |

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.8+ | Required for local installation |
| **GPU** | CUDA-compatible | Recommended for optimal performance |
| **RAM** | 16GB+ | 8GB minimum, 16GB recommended |

### Installation Options

#### Direct Installation

```bash
# Install dependencies
pip install unsloth transformers torch accelerate bitsandbytes

# Clone the repository
git clone https://github.com/noobezlol/Aletheia-Llama-3.2-3B
cd Aletheia-Llama-3.2-3B
```

#### Docker Setup

```bash
# Using Docker Compose (interactive mode)
docker compose up

# Or build manually
docker build -t llama32-uncensored .
docker run --gpus all -it --rm llama32-uncensored
```

## Usage

### Basic Usage

Run the chat interface directly:

```bash
python Final-chat.py
```

### Programmatic Usage

The main class is in `Final-chat.py` - run it directly. No import needed since it's designed as a standalone script.

To use programmatically, modify `Final-chat.py` or create a wrapper script that imports the `UncensoredChat` class.

### Advanced Configuration

```python
# Initialize with custom model path
chat = UncensoredChat(model_path="Ishaanlol/Aletheia-Llama-3.2-3B")

# Generate with custom parameters
response = chat.stream_response(
    "Tell me about AI", 
    max_new_tokens=1024, 
    temperature=0.7
)
```

### Code Style

<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin: 15px 0;">

<div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; color: #1a202c; font-weight: 500;">
Follow PEP 8 guidelines
</div>

<div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; color: #1a202c; font-weight: 500;">
Use type hints
</div>

<div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; color: #1a202c; font-weight: 500;">
Add docstrings to all functions
</div>

<div style="background: rgba(255,255,255,0.9); padding: 15px; border-radius: 8px; color: #1a202c; font-weight: 500;">
Write comprehensive tests
</div>

</div>

</div>

### Model Specifications

| Specification | Value |
|---------------|-------|
| **Architecture** | Llama 3.2-based |
| **Parameters** | 3 billion |
| **Context Length** | 8k (Native) / 128k (Supported) |
| **Precision** | FP16/BF16 support |
| **Quantization** | 4-bit & 8-bit available |

### Performance Benchmarks

<div align="center">

**Performance Comparison Charts**

![Performance Comparison](images/performance_comparison.png)

![Refusal Rate Comparison](images/refusal_rate_comparison.png)

**Key Performance Indicators**

| Metric | Score | Status |
|--------|-------|--------|
| Refusal Rate | 0% | Excellent |
| AdvBench Success | 12/12 | Perfect |
| HarmBench Success | 12/12 | Perfect |

</div>


## Portable GGUF (CPU/Mac/Ollama)

### Download Link

Download the GGUF file directly from Hugging Face:
https://huggingface.co/Ishaanlol/Aletheia-Llama-3.2-3B/blob/main/Llama-3.2-3B-Instruct.Q4_K_M.gguf

### Trade-off Warning

| Version | Intelligence | Stability | Requirements | Recommended Use |
|---------|-------------|-----------|--------------|-----------------|
| **Full Adapter** | Maximum Intelligence | 100% Stability | NVIDIA GPU Required | Complex coding, advanced reasoning, research tasks |
| **GGUF** | High Portability | ~5-10% Logic Degradation | CPU/Mac Compatible | Creative writing, text generation, general use |

**Important Note**: The 4-bit quantization on a small 3B model results in slight logic degradation for mathematical and complex reasoning tasks. However, the GGUF version maintains excellent performance for creative writing, content generation, and general text processing tasks.

### Usage Instructions

#### Ollama Setup

```bash
# Create the model with Ollama
ollama create aletheia-3b -f Modelfile

# Run the model
ollama run aletheia-3b
```

#### Python CPU Usage

For CPU-based execution without Ollama, use the included GGUF-chat.py script:

```bash
python GGUF-chat.py
```

This script provides the same uncensored functionality as the main adapter version but runs efficiently on CPU hardware.

## API Reference

### UncensoredChat Class

The main class is defined in `Final-chat.py`. To use programmatically, copy the class to your script or rename the file to `Final_chat.py`.

#### Methods

| Method | Description | Parameters |
|--------|-------------|------------|
| `__init__()` | Initialize the chat interface | `model_path` (str): Path to the model |
| `stream_response()` | Generate a streaming response | `user_input`, `max_new_tokens`, `temperature` |
| `clear_history()` | Clear conversation history | None |
| `save_conversation()` | Save conversation to file | `filename` (str, optional) |
| `show_help()` | Display help information | None |
| `run()` | Start the interactive chat | None |

#### Usage Examples

```python
# Direct usage - run the standalone script
python Final-chat.py

# To use programmatically, copy the UncensoredChat class 
# from Final-chat.py into your own script
```

## Docker Deployment

### Development Environment

Start the chat interface using Docker Compose:

```bash
docker compose up
```

The container will automatically start the chat interface.

**Container Details:**
- **Container Name**: uncensored-llama
- **Service Name**: llama-chat
- **GPU Access**: Automatically configured via docker compose
- **Model Cache**: Mounted to `~/.cache/huggingface` for persistent storage
- **Interactive**: Full terminal support with TTY and stdin
- **Entry Point**: Automatically launches Final-chat.py

### Manual Docker Build

Alternatively, build and run manually:

```bash
# Build the image
docker build -t llama32-uncensored .

# Run the chat interface
docker run --gpus all -it --rm llama32-uncensored

# Run with custom model path
docker run --gpus all -it --rm -e MODEL_PATH=Ishaanlol/Aletheia-Llama-3.2-3B llama32-uncensored
```

**Configuration Details:**

| Setting | Value |
|---------|-------|
| **Base Image** | unsloth/unsloth (includes CUDA and transformers) |
| **Container Name** | uncensored-llama |
| **Service Name** | llama-chat |
| **GPU Access** | Automatically configured via docker compose |
| **Model Cache** | Mounted to `~/.cache/huggingface` for persistent storage |
| **Interactive** | Full terminal support with TTY and stdin |
| **Entry Point** | Automatically launches Final-chat.py |

## Configuration

### Environment Variables

Configure the model and system behavior using environment variables:

#### Model Configuration
```bash
MODEL_PATH=Ishaanlol/Aletheia-Llama-3.2-3B
MAX_TOKENS=1024
TEMPERATURE=0.7
```

#### System Configuration
```bash
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4
```

### Model Parameters

Customize generation behavior with these parameters:

```python
{
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
    "pad_token_id": 50256
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 512 | Maximum tokens to generate |
| `temperature` | 0.7 | Controls randomness (0.0-2.0) |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `top_k` | 50 | Limit to top-k tokens |
| `repetition_penalty` | 1.1 | Penalize repetition |
| `do_sample` | True | Enable sampling |
| `pad_token_id` | 50256 | Padding token ID |

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/noobezlol/Aletheia-Llama-3.2-3B

# Install dependencies
pip install unsloth transformers torch accelerate bitsandbytes

# Run the chat interface
python Final-chat.py

# For programmatic usage, copy the UncensoredChat class 
# from Final-chat.py to your own script
```

### Code Style

We follow these guidelines to maintain code quality:

| Guideline | Description |
|-----------|-------------|
| **PEP 8** | Follow Python style guidelines |
| **Type Hints** | Use type annotations for better code clarity |
| **Docstrings** | Add comprehensive documentation to all functions |
| **Testing** | Write comprehensive tests for new features |

## Security Considerations

> **Important Security Notice**: These models are designed for research purposes. When deploying in production environments:

### Production Deployment Guidelines

| Guideline | Description |
|-----------|-------------|
| **Content Filtering** | Implement appropriate content filtering |
| **Output Monitoring** | Monitor model outputs for compliance |
| **Regulatory Compliance** | Ensure compliance with local regulations |
| **Ethical Considerations** | Consider the ethical implications of your use case |

### Security Analysis

![Comprehensive Security Analysis](images/comprehensive_security_analysis.png)

![Asymmetric Security](images/asymmetric_security.png)

![Security Implications](images/security_implications.png)

## License

This project is licensed under the **HIGH-RISK ARTIFICIAL INTELLIGENCE RESEARCH LICENSE (HAIR-L) Version 1.0** - see the [LICENSE](LICENSE) file for complete terms and conditions.

**IMPORTANT**: This is a strict liability shield license designed for AI safety research. By using this software, you acknowledge that you have read, understood, and agree to be bound by all terms in the LICENSE file.

## Acknowledgments

We acknowledge the following organizations and communities:

- **Meta AI** for the Llama 3.2 architecture
- **The open-source AI community** for research and development
- **Contributors** to the AI safety research community

## Support

For questions, issues, or contributions:

| Channel | Link |
|---------|------|
| **Email** | ishaanjeevan123@gmail.com |
| **Discord** | [Join our community](https://discord.gg/FU7RyMtK) |

---

<div align="center">

**Made with by the AI Research Community**

[Back to Top](#llama-32-uncensored-models)

</div>