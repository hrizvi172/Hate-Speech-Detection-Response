"""
pipeline.py
-----------
Unified pipeline that connects:
  1. Preprocessor     → cleans input text
  2. RoBERTa classifier → detects hate speech / offensive / neither
  3. DialoGPT generator → generates counterspeech if hate detected

Usage:
  from src.pipeline import HateSpeechPipeline
  pipeline = HateSpeechPipeline()
  result   = pipeline.run("some tweet text")
"""

import os
import sys
import torch
import numpy as np
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          AutoModelForCausalLM)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config      import MODELS_DIR, LABEL_NAMES, MAX_TOKEN_LENGTH
from src.preprocessor import clean_text


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, 'roberta_final')
CLASSIFIER_TOKENIZER  = os.path.join(MODELS_DIR, 'roberta_final')
GENERATION_MODEL_PATH  = os.path.join(MODELS_DIR, 'generation', 'dialogpt_best.pt')
GENERATION_TOKENIZER   = os.path.join(MODELS_DIR, 'generation', 'dialogpt_tokenizer')


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class HateSpeechClassifier:
    """
    Wraps the fine-tuned RoBERTa model.
    Classifies text into: Hate Speech / Offensive Language / Neither
    """

    def __init__(self, device: torch.device):
        self.device = device
        print("Loading classifier (RoBERTa)...")

        self.tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_MODEL_PATH
        )
        self.model = self.model.to(device)
        self.model.eval()
        print("  Classifier ready.")

    def predict(self, text: str) -> dict:
        """
        Returns predicted label, confidence, and all class probabilities.
        """
        cleaned = clean_text(text)
        if not cleaned.strip():
            cleaned = text  # fallback to raw if cleaning removes everything

        encoding = self.tokenizer(
            cleaned,
            max_length=MAX_TOKEN_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids      = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask)
            logits  = outputs.logits

        probs      = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))

        return {
            'label'      : pred_label,
            'label_name' : LABEL_NAMES[pred_label],
            'confidence' : float(probs[pred_label]),
            'probabilities': {
                LABEL_NAMES[i]: float(probs[i])
                for i in range(3)
            }
        }


# ══════════════════════════════════════════════════════════════════════════════
# GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class CounterSpeechGenerator:
    """
    Wraps the fine-tuned DialoGPT model.
    Generates counterspeech given a hateful text.
    Supports adjustable tone attributes.
    """

    TONE_PREFIXES = {
        'polite'     : "Respectfully, ",
        'informative': "It is important to know that ",
        'empathetic' : "I understand your concern, but ",
        'direct'     : "This is incorrect: ",
        'questioning': "Have you considered that ",
    }

    def __init__(self, device: torch.device):
        self.device = device
        print("Loading generator (DialoGPT)...")

        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_TOKENIZER)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base architecture
        self.model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')

        # Load fine-tuned weights BEFORE resizing embeddings
        state_dict = torch.load(GENERATION_MODEL_PATH, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)

        # Resize AFTER loading weights
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = self.model.to(device)
        self.model.eval()
        print("  Generator ready.")

    def generate(self,
             hate_speech : str,
             tone        : str   = 'polite',
             max_tokens  : int   = 80,
             temperature : float = 0.7,
             top_p       : float = 0.9) -> str:

        prefix = self.TONE_PREFIXES.get(tone, "Respectfully, ")

        # Richer prompt gives the model more context to continue from
        input_text = (
            f"Hateful comment: {hate_speech}\n"
            f"Counterspeech response: {prefix}"
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=150
        ).to(self.device)

        with torch.no_grad():
            try:
                output_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            except RuntimeError:
                output_ids = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        response   = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        if prefix and not response.startswith(prefix.strip()):
            response = prefix + response

        # Fallback responses per tone if model generates nothing useful
        fallbacks = {
            'polite'     : f"{prefix}every person deserves to be treated with dignity regardless of their background.",
            'informative': f"{prefix}research shows that diversity strengthens communities rather than weakening them.",
            'empathetic' : f"{prefix}I hear your frustration, though generalizing about groups of people causes real harm.",
            'direct'     : f"{prefix}generalizing about entire groups of people is factually wrong and harmful.",
            'questioning': f"{prefix}the evidence actually shows the opposite of what you're suggesting?",
        }

        if not response or len(response.replace(prefix.strip(), '').strip()) < 5:
            response = fallbacks.get(tone, fallbacks['polite'])

        return response


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class HateSpeechPipeline:
    """
    Full pipeline: text → classify → (if hate) → generate counterspeech

    Example
    -------
    pipeline = HateSpeechPipeline()
    result   = pipeline.run("your tweet here", tone="polite")
    print(result)
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Pipeline using device: {self.device}\n")

        self.classifier = HateSpeechClassifier(self.device)
        self.generator  = CounterSpeechGenerator(self.device)
        print("\nPipeline ready.\n")

    def run(self,
            text        : str,
            tone        : str   = 'polite',
            temperature : float = 0.7,
            top_p       : float = 0.9,
            max_tokens  : int   = 80) -> dict:
        """
        Run the full pipeline on a single text input.

        Returns a dict with:
          - original_text
          - cleaned_text
          - classification (label, confidence, all probabilities)
          - counterspeech (only if hate speech detected)
          - action_taken
        """
        result = {
            'original_text' : text,
            'cleaned_text'  : clean_text(text),
            'classification': None,
            'counterspeech' : None,
            'action_taken'  : None
        }

        # Step 1: Classify
        classification = self.classifier.predict(text)
        result['classification'] = classification

        # Step 2: Act based on label
        label = classification['label']

        if label == 0:   # Hate Speech
            result['action_taken'] = 'counterspeech_generated'
            result['counterspeech'] = self.generator.generate(
                text,
                tone=tone,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )

        elif label == 1:  # Offensive Language
            result['action_taken'] = 'flagged_for_review'

        else:             # Neither
            result['action_taken'] = 'no_action_needed'

        return result

    def run_batch(self, texts: list, tone: str = 'polite') -> list:
        """Run pipeline on a list of texts."""
        return [self.run(t, tone=tone) for t in texts]

    def format_result(self, result: dict) -> str:
        """Pretty-print a pipeline result."""
        lines = [
            "=" * 60,
            f"INPUT    : {result['original_text'][:100]}",
            f"CLEANED  : {result['cleaned_text'][:100]}",
            "-" * 60,
            f"LABEL    : {result['classification']['label_name']}",
            f"CONFIDENCE: {result['classification']['confidence']:.1%}",
            "",
            "PROBABILITIES:"
        ]
        for name, prob in result['classification']['probabilities'].items():
            bar = '█' * int(prob * 20)
            lines.append(f"  {name:20s} {prob:.1%} {bar}")

        lines.append(f"\nACTION   : {result['action_taken']}")

        if result['counterspeech']:
            lines.append(f"\nCOUNTER  : {result['counterspeech']}")

        lines.append("=" * 60)
        return '\n'.join(lines)