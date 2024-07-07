# Spam Detection Module

This repository contains the implementation of the spam detection module for our cybersecurity platform. The module is based on the methodology introduced in the JPMorgan paper ["Spam-T5: Benchmarking Large Language Models for Few-Shot Email Spam Detection" (Labonne & Moran, 2023)](https://arxiv.org/abs/2304.01238). This paper explores the effectiveness of large language models (LLMs) in email spam detection and introduces Spam-T5, a fine-tuned version of Google's Flan-T5 model specifically adapted for this purpose.

## Overview

We reproduced the model by referencing JPMorgan’s open-sourced work on [GitHub]](https://github.com/jpmorganchase/llm-email-spam-detection), with some modifications. Instead of the datasets used by JPMorgan, we employed the Enron 2006, TREC 2005, TREC 2006, and TREC 2007 datasets retrieved through the Kaggle API. 

## Performance Metrics

The performance of our spam detection module is summarized in the table below:

| Dataset  | F1 Score | Precision | Recall | Accuracy | Training Time (s) | Inference Time (s) |
|----------|----------|-----------|--------|----------|-------------------|-------------------|
| Enron    | 0.992    | 0.989     | 0.994  | 0.992    | 4960              | 49                |
| TREC 2005| 0.987    | 0.989     | 0.986  | 0.989    | 9197              | 87                |
| TREC 2006| 0.991    | 0.990     | 0.992  | 0.995    | 2853              | 28                |
| TREC 2007| 0.995    | 0.997     | 0.994  | 0.995    | 8771              | 85                |
| **Average** | **0.991** | **0.991** | **0.992** | **0.993** | **6445** | **62** |

## Model
Our trained model is made available on [HuggingFace](https://huggingface.co/periscopehf/capstone-email-classifier).

## References

- Maxime Labonne, Sean Moran ["Spam-T5: Benchmarking Large Language Models for Few-Shot Email Spam Detection"](https://arxiv.org/abs/2304.01238)
- [GitHub Repository for Spam-T5](https://github.com/jpmorganchase/llm-email-spam-detection)
- [Enron 2006 dataset](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-enron-2006?select=email_text.csv)
- [Trec 2005 dataset](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2005?select=email_text.csv)
- [Trec 2006 dataset](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2006?select=email_text.csv)
- [Trec 2007 dataset](https://www.kaggle.com/datasets/bayes2003/emails-for-spam-or-ham-classification-trec-2007?select=email_text.csv)


