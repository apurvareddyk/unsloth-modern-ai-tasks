# Modern AI Fine-Tuning, Reinforcement Learning, and Pretraining with Unsloth.ai

## Project Overview
This repository contains Google Colab notebooks and YouTube tutorials demonstrating **modern AI workflows** using the **Unsloth.ai** framework.  
The project showcases how to fine-tune, align, and extend small open-weight language models such as **SmolLM2-135M** efficiently on limited compute.  

The assignment covers **five key Colabs**, each demonstrating a core modern LLM technique:
- **A:** Full Fine-Tuning on a small model (**SmolLM2-135M**) for a chosen task such as coding or chat.  
- **B:** LoRA Parameter-Efficient Fine-Tuning of the same model and dataset.  
- **C:** Reinforcement Learning with Preference Data (**DPO/RLHF**) — training on preferred vs. rejected responses.  
- **D:** **GRPO-Based Reasoning Reinforcement Learning** — self-improving reasoning model.  
- **E:** Continued Pretraining — teaching the model a new language or domain.  

---

## Models Demonstrated
This project primarily demonstrates **SmolLM2-135M**, an ultra-lightweight model suited for full fine-tuning and LoRA training.  

Additional open-weight models referenced for exploration include:
- **Gemma-3-1B-IT (Unsloth 4-bit)** – small chat/instruction model.  
- **Llama 3 / 3.1 (8B)** – for higher-capacity instruction following.  
- **Phi-3 / 3.5 (Mini)** – efficient chat model used for alignment and reward learning.  
- **TinyLlama (1.1B)** – fast experimentation and multilingual pretraining.  
- **Qwen2 (7B)** and **Mistral v0.3 (7B)** – for advanced experiments and transfer comparison.  

---

## Colab Notebooks & Demonstrations

Click below to open the Google Colab notebooks for each assignment part:

* **A: Full Fine-Tuning (SmolLM2-135M)**  
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O9drCJgEqdqQPElHi3jfwHVB58ppNoC8?usp=sharing)

* **B: LoRA Parameter-Efficient Fine-Tuning (SmolLM2-135M)**  
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LHYbRvcglCDNzdTfoNpOJzaM0dZyauw3?usp=sharing)

* **C: Reinforcement Learning with Preference Data (DPO/RLHF)**  
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1du0T0Ekm5KPd8keHPPSAXbWW_Zyn-GeM?usp=sharing)

* **D: GRPO-Based Reasoning Reinforcement Learning**  
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SgGtJrViD-fPXnYiZsAlld48egP5T51a?usp=sharing)

* **E: Continued Pretraining (Language Adaptation)**  
  * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dwbOaaTijWD4dFQQWTNpfcoHTAZRqAtg?usp=sharing)

---

## Key Features Demonstrated

### Full Fine-Tuning (Colab 1)
- Fine-tunes the entire **SmolLM2-135M** model (`full_finetuning=True`) on a small dataset such as **CodeAlpaca** or **chat-instruction** data.  
- Demonstrates complete weight updates, explaining dataset formatting, chat templates, and evaluation.

### LoRA Fine-Tuning (Colab 2)
- Uses **LoRA adapters** for parameter-efficient training on the same dataset.  
- Shows how LoRA reduces memory usage while maintaining accuracy.  
- Includes before/after inference comparisons and explanation of LoRA parameters (`r`, `alpha`, and `dropout`).

### Reinforcement Learning with Preference Data (Colab 3)
- Implements **Direct Preference Optimization (DPO)** using datasets containing *chosen* and *rejected* outputs.  
- Explains reward alignment and preference-based optimization.  
- Demonstrates how reinforcement signals refine model behavior.

### GRPO Reasoning Reinforcement Learning (Colab 4)
- Trains a reasoning model using **GRPO**, where rewards are derived from model-generated answers.  
- Focuses on logical or mathematical reasoning tasks.  
- Explains the concept of self-play and automatic reward functions.

### Continued Pretraining (Colab 5)
- Extends model capabilities via **continued pretraining** on raw text corpora in a new language or domain.  
- Shows how to adapt **SmolLM2-135M** to new vocabulary patterns efficiently.  
- Demonstrates text generation in the newly learned language.

---

## Video Tutorials
Watch detailed video walkthroughs for every part of the assignment:  
[![YouTube Playlist](https://img.shields.io/badge/YouTube-Watch_Tutorials-red?logo=youtube)](https://www.youtube.com/playlist?list=PLGHkLcp2I_S8ok6Wdj_tDTilvYX1zohvj)

---

## Usage Workflow Overview

### 1️⃣ Full Fine-Tuning (Colab 1)
- Load **SmolLM2-135M** and enable `full_finetuning=True`.  
- Prepare a small instruction-following dataset (e.g., CodeAlpaca).  
- Run **SFTTrainer** with Unsloth for end-to-end fine-tuning.  
- Record and explain model behavior and loss curve.

### 2️⃣ LoRA Fine-Tuning (Colab 2)
- Load the same base model in **4-bit** mode.  
- Attach **LoRA adapters** and run parameter-efficient training.  
- Show faster convergence and lower VRAM usage.

### 3️⃣ RL with Preference Data (Colab 3)
- Load datasets like **StackExchange Paired** or **Anthropic HH**.  
- Train using **DPOTrainer** to align model preferences with human-preferred responses.

### 4️⃣ GRPO Reasoning (Colab 4)
- Use problem-solving datasets (e.g., **GSM8K**).  
- Generate and score answers automatically with reward functions.  
- Train with **GRPOTrainer** to improve reasoning ability.

### 5️⃣ Continued Pretraining (Colab 5)
- Use Unsloth’s **SFTTrainer** for unsupervised LM pretraining on new text.  
- Demonstrate multilingual or domain-specific adaptation.  
- Optionally export to **Ollama** and show inference.

---

## Tips & Best Practices
- **Choose small models** (≤1B parameters) like **SmolLM2-135M** for faster experimentation.  
- **Use chat templates** (`tokenizer.apply_chat_template`) for consistent formatting.  
- Set `packing=True` to reduce training steps on short samples.  
- Enable **gradient checkpointing** for memory savings.  
- Use **4-bit quantization + LoRA** when GPU memory is limited.  
- Prefer **DPO or GRPO** for aligning models to preference or reasoning tasks.  
- For continued pretraining, start from a checkpoint and fine-tune on new language text.  
- Follow **ethical practices** when fine-tuning on sensitive domains.

---

## References
- [Unsloth Documentation](https://docs.unsloth.ai)  
- [Fine-Tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide)  
- [Reinforcement Learning Guide (DPO/GRPO)](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)  
- [Continued Pretraining Docs](https://docs.unsloth.ai/basics/continued-pretraining)  
- [Medium – Unsloth LoRA + Ollama Tutorial](https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-lightweight-solution-to-full-cycle-llm-development-edadb6d9e0f0)  
- [Unsloth Notebooks on GitHub](https://github.com/unslothai/notebooks)  
- [Kaggle Example – Fine-Tuning LLMs Using Unsloth](https://www.kaggle.com/code/kingabzpro/fine-tuning-llms-using-unsloth)  
- [Unsloth Blog – R1 Reasoning](https://unsloth.ai/blog/r1-reasoning)
