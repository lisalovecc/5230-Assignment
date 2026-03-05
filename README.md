ST5230 Assignment 1: Language Models and Representations

This repository contains the implementation for Assignment 1 of ST5230 (Applied Natural Language Processing). The project explores various neural language model architectures, conducts embedding ablation studies, and investigates the impact of self-supervised representation learning on downstream NLP tasks.

All experiments are conducted on the IMDB Large Movie Review Dataset.

📁 Repository Structure

The project is divided into three main parts, corresponding to the assignment requirements:

part1_language_models.py

Implements and compares four different language modeling architectures: N-gram (n=3), Vanilla RNN, LSTM, and Transformer.

Evaluates models based on training/inference time, validation perplexity (PPL), and qualitative text generation.

part2_embedding_ablation.py

Investigates the impact of distributed representations.

Compares three embedding strategies within an LSTM language model:

Trainable embeddings from scratch (Unfrozen).

Self-trained Custom Word2Vec CBOW embeddings (Frozen).

Simulated Public GloVe embeddings (Frozen).

part3_downstream_task.py

Evaluates the utility of learned representations on a downstream Binary Sentiment Classification task.

Includes a Data Scaling Ablation loop (evaluating at 1k, 3k, 5k, and 10k reviews) to demonstrate how pre-trained Language Model embeddings improve sample efficiency compared to learning from scratch (Baseline).

🚀 Requirements and Setup

To run the scripts, ensure you have Python 3.8+ and the following dependencies installed:

pip install torch torchvision torchaudio
pip install datasets


Note: The code is highly optimized for GPU execution. A CUDA-enabled GPU (e.g., NVIDIA T4) is strongly recommended for reproducing the full-scale experiments in Part III.

⚙️ How to Run

Each Python script is completely self-contained and handles its own data loading, preprocessing, training, and evaluation. Simply execute them from the terminal:

python part1_language_models.py
python part2_embedding_ablation.py
python part3_downstream_task.py


📊 Key Highlights & Findings

Architecture Limits: LSTMs effectively solve the vanishing gradient problem observed in vanilla RNNs over sequence lengths of 35, achieving lower perplexity. Transformers demonstrate superior training speed due to self-attention parallelization.

Domain Mismatch: Task-specific embeddings (even when trained from scratch) naturally outperform generic public embeddings (like GloVe) which suffer from domain mismatch on cinematic/slang vocabulary.

Positive Transfer & Sample Efficiency: Transferring embeddings from a pre-trained language model provides a significant "warm start" for downstream tasks. The performance gap is most prominent in low-resource regimes (e.g., +5.00% accuracy improvement at 1,000 reviews), proving that next-token prediction inherently captures deep, transferable semantic knowledge.

Author: Yiqiu Pan
