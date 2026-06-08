"""
app.py
------
Gradio web demo for the Hate Speech Detection + Counterspeech pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from src.pipeline import HateSpeechPipeline

print("Initializing pipeline...")
pipeline = HateSpeechPipeline()
print("Ready.\n")

# Store last classified text and tone between button clicks
_last = {"text": "", "tone": "polite"}


def classify_only(text, tone):
    if not text.strip():
        return "⚠️ Please enter some text.", "", "", "Run classification first."

    _last["text"] = text
    _last["tone"] = tone

    clf = pipeline.classifier.predict(text)

    label_map = {
        'Hate Speech'        : '🚨 Hate Speech',
        'Offensive Language' : '⚠️ Offensive Language',
        'Neither'            : '✅ Neither',
    }
    label      = label_map[clf['label_name']]
    confidence = f"{clf['confidence']:.1%}"

    probs = clf['probabilities']
    prob_text = "\n".join([
        f"{'🚨 Hate Speech':<25} {probs['Hate Speech']:.1%}",
        f"{'⚠️ Offensive Language':<25} {probs['Offensive Language']:.1%}",
        f"{'✅ Neither':<25} {probs['Neither']:.1%}",
    ])

    if clf['label'] == 0:
        counter_placeholder = "⏳ Hate speech detected — click 'Generate Counterspeech' when ready."
    elif clf['label'] == 1:
        counter_placeholder = "⚠️ Flagged as offensive. Click 'Generate Counterspeech' for a response."
    else:
        counter_placeholder = "✅ No counterspeech needed for this input."

    return label, confidence, prob_text, counter_placeholder


def generate_counter():
    text = _last.get("text", "")
    tone = _last.get("tone", "polite")

    if not text:
        return "⚠️ Please run classification first."

    clf = pipeline.classifier.predict(text)
    if clf['label'] == 2:
        return "✅ This text was classified as Neither — no counterspeech needed."

    return pipeline.generator.generate(text, tone=tone)


# ── Build UI ──────────────────────────────────────────────
with gr.Blocks(title="Hate Speech Detector", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🛡️ Hate Speech Detection & Counterspeech Generator
    **Step 1:** Enter text and click **Analyze** for instant classification.
    **Step 2:** If hate/offensive content detected, click **Generate Counterspeech**.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Type or paste text here...",
                lines=4
            )
            tone_input = gr.Radio(
                choices=['polite', 'informative', 'empathetic', 'direct', 'questioning'],
                value='polite',
                label="Counterspeech Tone"
            )
            with gr.Row():
                analyze_btn  = gr.Button("⚡ Analyze",             variant="primary")
                generate_btn = gr.Button("💬 Generate Counterspeech", variant="secondary")

        with gr.Column(scale=2):
            label_out      = gr.Textbox(label="Classification")
            confidence_out = gr.Textbox(label="Confidence")
            probs_out      = gr.Textbox(label="Class Probabilities", lines=4)
            counter_out    = gr.Textbox(label="Generated Counterspeech", lines=3)

    gr.Examples(
        examples=[
            ["I hate all people from that country", "polite"],
            ["These people are ruining everything", "informative"],
            ["Have a wonderful day everyone!", "empathetic"],
            ["They should all go back where they came from", "questioning"],
        ],
        inputs=[text_input, tone_input]
    )

    analyze_btn.click(
        fn=classify_only,
        inputs=[text_input, tone_input],
        outputs=[label_out, confidence_out, probs_out, counter_out]
    )

    generate_btn.click(
        fn=generate_counter,
        inputs=[],
        outputs=[counter_out]
    )

if __name__ == "__main__":
    demo.launch(share=False)


# """
# app.py
# ------
# Gradio web demo for the Hate Speech Detection + Counterspeech pipeline.
# Run with: python demo/app.py
# """

# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import gradio as gr
# from src.pipeline import HateSpeechPipeline

# # Load pipeline once at startup
# print("Initializing pipeline...")
# pipeline = HateSpeechPipeline()
# print("Ready.\n")


# def analyze(text, tone):
#     if not text.strip():
#         return "⚠️ Please enter some text.", "", "", ""

#     result = pipeline.run(text, tone=tone)
#     clf    = result['classification']

#     # Label with emoji
#     label_map = {
#         'Hate Speech'        : '🚨 Hate Speech',
#         'Offensive Language' : '⚠️ Offensive Language',
#         'Neither'            : '✅ Neither',
#     }
#     label      = label_map[clf['label_name']]
#     confidence = f"{clf['confidence']:.1%}"

#     # Probability breakdown
#     probs = clf['probabilities']
#     prob_text = "\n".join([
#         f"{'🚨 Hate Speech':<25} {probs['Hate Speech']:.1%}",
#         f"{'⚠️ Offensive Language':<25} {probs['Offensive Language']:.1%}",
#         f"{'✅ Neither':<25} {probs['Neither']:.1%}",
#     ])

#     # Counterspeech
#     counter = result['counterspeech'] or "No counterspeech needed for this input."

#     return label, confidence, prob_text, counter


# # ── Build UI ──────────────────────────────────────────────
# with gr.Blocks(title="Hate Speech Detector", theme=gr.themes.Soft()) as demo:

#     gr.Markdown("""
#     # 🛡️ Hate Speech Detection & Counterspeech Generator
#     Enter any text to classify it and generate an appropriate counterspeech response.
#     """)

#     with gr.Row():
#         with gr.Column(scale=2):
#             text_input = gr.Textbox(
#                 label="Input Text",
#                 placeholder="Type or paste text here...",
#                 lines=4
#             )
#             tone_input = gr.Radio(
#                 choices=['polite', 'informative', 'empathetic', 'direct', 'questioning'],
#                 value='polite',
#                 label="Counterspeech Tone"
#             )
#             submit_btn = gr.Button("Analyze", variant="primary")

#         with gr.Column(scale=2):
#             label_out      = gr.Textbox(label="Classification")
#             confidence_out = gr.Textbox(label="Confidence")
#             probs_out      = gr.Textbox(label="Class Probabilities", lines=4)
#             counter_out    = gr.Textbox(label="Generated Counterspeech", lines=3)

#     gr.Examples(
#         examples=[
#             ["I hate all people from that country", "polite"],
#             ["These people are ruining everything", "informative"],
#             ["Have a wonderful day everyone!", "empathetic"],
#             ["This policy is completely wrong", "direct"],
#             ["They should all go back where they came from", "questioning"],
#         ],
#         inputs=[text_input, tone_input]
#     )

#     submit_btn.click(
#         fn=analyze,
#         inputs=[text_input, tone_input],
#         outputs=[label_out, confidence_out, probs_out, counter_out]
#     )

# if __name__ == "__main__":
#     demo.launch(share=False)