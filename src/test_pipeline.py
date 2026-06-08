import sys
sys.path.insert(0, 'src')
from pipeline import HateSpeechPipeline

pipeline = HateSpeechPipeline()

tests = [
    ('I hate all people from that country', 'polite'),
    ('Have a great day everyone!', 'informative'),
    ('This is total garbage', 'empathetic'),
]

for text, tone in tests:
    result = pipeline.run(text, tone=tone)
    print(pipeline.format_result(result))
    print()