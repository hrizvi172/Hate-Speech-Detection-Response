# 🛡️ Hate Speech Detection & Counterspeech Generation

An end-to-end NLP pipeline that automatically detects hate speech in text and generates contextually appropriate counterspeech responses. Built with a fine-tuned **RoBERTa** classifier and a fine-tuned **DialoGPT** generator, deployed as an interactive **Gradio** web application.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Technologies](#technologies)

---

## Overview

Online hate speech causes real psychological harm and is too prevalent for human moderation alone. This project builds a two-stage automated system:

1. **Classification** — A fine-tuned RoBERTa model classifies text into three categories: *Hate Speech*, *Offensive Language*, or *Neither*
2. **Generation** — A fine-tuned DialoGPT model generates a counterspeech response across five tone modes when hate speech is detected

The system routes each input intelligently:
- **Hate Speech** → generate counterspeech
- **Offensive Language** → flag for moderator review
- **Neither** → pass through, no action

---

## Results

### Classifier Performance (Davidson test set)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 0.812 | 0.734 | 0.701 | 0.717 |
| Logistic Regression | 0.876 | 0.821 | 0.798 | 0.809 |
| SVM | 0.889 | 0.843 | 0.812 | 0.827 |
| Random Forest | 0.882 | 0.831 | 0.805 | 0.818 |
| LSTM + GloVe | 0.901 | 0.867 | 0.841 | 0.854 |
| **RoBERTa (ours)** | **0.923** | **0.891** | **0.876** | **0.883** |

### Counterspeech Tone Examples

For input: *"I hate all people from that country"*

| Tone | Generated Response |
|---|---|
| Polite | Respectfully, every person deserves to be treated with dignity regardless of their background. |
| Informative | It is important to know that research shows diversity strengthens communities. |
| Empathetic | I understand your concern, but generalizing about groups of people causes real harm. |
| Direct | This is incorrect: generalizing about entire groups of people is factually wrong and harmful. |
| Questioning | Have you considered that the evidence actually shows the opposite of what you're suggesting? |

---

## Project Structure

```
hate-speech-project/
│
├── src/
|   ├── analyze_features.py       # Script/utilities to analyze extracted features
|   ├── config.py                 # Paths, constants, and global configuration settings
|   ├── explore_data.py           # Exploratory data analysis (EDA) and data inspection
|   ├── feature_extractor.py      # Core logic for feature extraction engineering
|   ├── pipeline.py               # Unified data and modeling pipeline
|   ├── prepare_conan.py          # Data preparation specific to the CONAN dataset
|   ├── prepare_data.py           # General dataset preparation and formatting
|   ├── preprocessor.py           # Text cleaning and preprocessing pipeline
|   ├── run_feature_extraction.py # Execution script to run feature extraction
|   ├── run_preprocessing.py      # Execution script to run data preprocessing
|   ├── test_pipeline.py          # Unit tests or validation scripts for the pipeline
|   ├── train_classical.py        # Training script for classical machine learning models
|   └── train_tfidf2048.py        # Training script using TF-IDF with a 2048 feature limit
│
├── notebooks/                 # Colab training notebooks
│   ├── train_classifier.ipynb
│   └── train_generator.ipynb
│
├── demo/
│   └── app.py                 # Gradio web application
│
├── models/                    # ← not tracked by git (see Models section)
│   ├── roberta_final/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   └── tokenizer.json
│   └── generation/
│       ├── dialogpt_best.pt
│       └── dialogpt_tokenizer/
│
├── data/                      # ← not tracked by git (see Dataset section)
│   ├── raw/
│   └── processed/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- 4GB+ RAM (8GB recommended for generation)
- GPU optional but speeds up generation significantly

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/hrizvi172/Hate-Speech-Detection-Response.git
cd Hate-Speech-Detection-Response

# 2. Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Models

The trained model files are not stored in this repository due to their size. The train scripts are added in /notebooks/ directory.
Now to trained them
> **[Download the davidson datasets from https://github.com/t-davidson/hate-speech-and-offensive-language.git]**
#
> **[Download the CONAN datasets from https://github.com/marcoguerini/CONAN.git]**

After downloading, train the models using the scripts in /notebooks/ directory
---

## Usage

### Run the Gradio Web App

```bash
python demo/app.py
```

Open your browser at `http://127.0.0.1:7860`

**Step 1:** Enter text and click **⚡ Analyze** — get instant classification  
**Step 2:** Click **💬 Generate Counterspeech** if hate speech is detected

### Use the Pipeline in Python

```python
from src.pipeline import HateSpeechPipeline

pipeline = HateSpeechPipeline()

result = pipeline.run("I hate all people from that country", tone="polite")
print(pipeline.format_result(result))
```

Output:
```
============================================================
INPUT    : I hate all people from that country
CLEANED  : hate people country
------------------------------------------------------------
LABEL    : Hate Speech
CONFIDENCE: 44.2%

PROBABILITIES:
  Hate Speech          44.2% ████████
  Offensive Language   25.9% █████
  Neither              29.9% █████

ACTION   : counterspeech_generated

COUNTER  : Respectfully, every person deserves to be treated with dignity regardless of their background.
============================================================
```

### Classify Only (faster)

```python
from src.pipeline import HateSpeechClassifier
import torch

clf = HateSpeechClassifier(torch.device('cpu'))
result = clf.predict("Have a great day everyone!")
# {'label': 2, 'label_name': 'Neither', 'confidence': 0.972, ...}
```

### Available Tone Modes

```python
tones = ['polite', 'informative', 'empathetic', 'direct', 'questioning']
result = pipeline.run("your text here", tone="empathetic")
```

---

## Models

### RoBERTa Classifier

- **Base model:** `roberta-base` (125M parameters)
- **Training data:** Davidson hate speech dataset (24,802 tweets)
- **Training:** 4 epochs, AdamW optimizer, lr=2e-5, batch size=16
- **Output:** 3-class (Hate Speech / Offensive Language / Neither)
- **Trained on:** Google Colab T4 GPU

### DialoGPT Generator

- **Base model:** `microsoft/DialoGPT-medium` (345M parameters)
- **Training data:** CONAN counter-narrative dataset (4,078 pairs)
- **Training:** 4 epochs, AdamW optimizer, lr=5e-5, batch size=16
- **Inference:** Beam search (4 beams), no-repeat n-gram (n=3)
- **Trained on:** Google Colab T4 GPU

---

## Dataset

### Davidson et al. (2017) — Hate Speech Detection

- 24,802 tweets annotated via CrowdFlower
- Labels: Hate Speech (5.8%), Offensive Language (77.4%), Neither (16.8%)
- Download: [GitHub](https://github.com/t-davidson/hate-speech-and-offensive-language)

### CONAN — Counter Narratives

- 4,078 hate speech and counter-narrative pairs
- Focused on anti-Muslim hate speech
- Download: [GitHub](https://github.com/marcoguerini/CONAN)

Place raw data files in `data/raw/` before running training notebooks.

---

## Architecture

```
Input Text
    │
    ▼
Preprocessor (URL/mention/HTML removal)
    │
    ▼
RoBERTa Classifier (3-class)
    │
    ├── Hate Speech ──► DialoGPT Generator ──► Tone Control ──► Counterspeech
    │
    ├── Offensive ────► Flagged for Review
    │
    └── Neither ──────► No Action
    │
    ▼
Pipeline Output (label + confidence + counterspeech)
    │
    ▼
Gradio Web App
```

---

## Technologies

| Component | Technology |
|---|---|
| Classifier | RoBERTa (HuggingFace Transformers) |
| Generator | DialoGPT-medium (HuggingFace Transformers) |
| Training | PyTorch, Google Colab T4 GPU |
| Preprocessing | regex, HuggingFace tokenizers |
| Web UI | Gradio |
| Baselines | scikit-learn, numpy |

---

## References

- Davidson et al. (2017). *Automated hate speech detection and the problem of offensive language.* ICWSM.
- Liu et al. (2019). *RoBERTa: A robustly optimized BERT pretraining approach.* arXiv:1907.11692.
- Zhang et al. (2020). *DialoGPT: Large-scale generative pre-training for conversational response generation.* ACL.
- Fanton et al. (2021). *Human-in-the-loop for data collection: A multi-target counter narrative dataset.* ACL-IJCNLP.

---
