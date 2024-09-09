# FineZip: README

## Overview
**FineZip** is a novel approach to lossless text compression using Large Language Models (LLMs). Building on previous work like LLMZip, FineZip pushes the boundaries of text compression by integrating both **online memorization** and **dynamic context size** techniques. These innovations lead to significant improvements in compression speed while maintaining competitive compression ratios compared to both traditional methods (e.g., gzip, bzip2) and neural network-based methods (e.g., NNCP, LLMZip). FineZip compresses text 54 times faster than LLMZip with a minor loss in compression performance.

### Main Contributions:
1. **FineZip** combines "online" memorization using parameter-efficient fine-tuning (PEFT) and "offline" pre-trained LLMs for text compression, enabling faster compression without sacrificing too much performance.
2. A **dynamic context window** allows batching of compression steps, significantly improving the compression speed.
3. **Quantization techniques** further optimize performance by reducing memory requirements, allowing larger batch sizes and faster compression times.

## File Structure
Here's an overview of the files included in this codebase:

- **`finezip/eval.py`**:  
  - The main script implementing FineZip compression using LLM-based techniques with online memorization and dynamic context handling. 
  - creates a **ZipModel** that implements the zipping/unzipping
  - function named **memory_eval** that allows for testing with multiple kinds of models (both base and finetuned), context sizes, and batch sizes

- **`finezip/finetune.py`**
  - The script that allows you to finetune models using LoRA and QLoRA
  - allows you to finetune multiple models at a time with different hyperparams

