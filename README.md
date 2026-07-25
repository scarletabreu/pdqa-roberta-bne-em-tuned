# PDQA RoBERTa BNE EM Tuned

Fine-tuning of a Spanish RoBERTa model for extractive Question Answering on the PDQA dataset.

## Overview

This repository contains the notebook, code and documentation used to fine-tune a RoBERTa model for the Dominican News Question Answering Challenge.

The project includes:

- Data preprocessing
- Fine-tuning of a RoBERTa model
- Hyperparameter comparison
- Model evaluation using Exact Match (EM) and F1
- Generation of predictions for the test set

## Hugging Face Model

The trained model is available on Hugging Face:

https://huggingface.co/scarletabreu/pdqa-roberta-bne-em-tuned

## Repository structure

```
.
├── 03_Introduccion_QA_Noticias.ipynb  # Notebook principal con el pipeline completo
├── requirements.txt                   # Dependencias del proyecto
└── LICENSE                            # Licencia Apache 2.0
```

## Dataset

- https://huggingface.co/datasets/Lisibonny/pdqa

## Requirements

```bash
pip install -r requirements.txt
````

## Authors

* Scarlet Abreu
* Renso Peralta

## License

Apache License 2.0.
