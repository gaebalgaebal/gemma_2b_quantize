# Project for 2-bit quantizing for distributing LLMs on edge devices. #
Testing with gemma-2b model(https://huggingface.co/google/gemma-2b) and (https://huggingface.co/shuyuej/gemma-2b-GPTQ)


Gemma-2B Quantization Experiment Project

Experimental Environment

Python version: 3.10

OS: Windows 11

Environment manager: Anaconda

GPU: NVIDIA RTX 4060 Ti (Note: INT2 quantized models run on CPU only in this setup)

Required Python libraries: torch, transformers, datasets, tqdm, auto-gptq

This project focuses on comparing the performance of the Gemma-2B language model under different quantization levels: FP16 (baseline), INT4 (GPTQ), and INT2 (custom configuration). INT2 quantization is currently limited to CPU execution due to CUDA kernel limitations in the prebuilt AutoGPTQ package on Windows.

Step-by-Step Instructions to Run the Experiments

1. Set Up the Python Virtual Environment

Using Anaconda:

conda create -n gptq_env python=3.10
conda activate gptq_env

Install the required packages:

pip install torch transformers datasets tqdm
pip install auto-gptq  # Install prebuilt version (Windows-compatible)

Note: If running on a GPU-enabled Linux or WSL environment, you may instead install AutoGPTQ from source to enable CUDA support.

2. INT2 Quantization of the Model

Run the quantization script:

cd gemma_project/scripts
python int2_quantize.py

This script will:

Load the original FP16 Gemma-2B model.

Apply 2-bit weight quantization using manually defined config parameters.

Save the quantized model in gemma-2b-INT2-GPTQ/.

You should see output indicating progress through 18 layers of quantization. Upon completion, the following files will be created:

quantized_model.safetensors

quantize_config.json

config.json, tokenizer.json, and related files

3. Inference and Perplexity Evaluation on INT2 Model

To run inference and evaluate language model performance using perplexity:

python int2_inference.py

This script will:

Load the quantized INT2 model and tokenizer.

Generate text based on a sample prompt.

Measure and report inference time.

Compute perplexity using the Wikitext-2 dataset (subset).

Note: Since this is executed on CPU, inference may be significantly slower than GPU-based runs.

4. Additional Experiments and Comparisons

Other scripts provided in this project allow evaluation of different quantization levels:

Script

Description

baseline.py

Runs the original FP16 model on GPU

int4_gptq.py

Runs the 4-bit quantized model using GPTQ

int2_inference.py

Runs the 2-bit quantized model on CPU

baseline_vs_int4_plot.py

Plots perplexity and performance comparison charts

Future Plans

Porting INT2 model to GPU using Linux or WSL2 and building AutoGPTQ from source

Adding quantized model evaluation on downstream tasks

Visualizing detailed performance metrics

Finalizing presentation materials for academic report or demo
