# in progress
from .extracted_features_dataset import ExtractedFeaturesDataset

data = ExtractedFeaturesDataset(
        path="/store/store4/data/LibriSpeech-Clean-NoSil/features/mimi/",
        split="valid",
        labels="wrd",
        )

char_dict = {}

for item in data:
    chars = item["target"]
    speech_toks = item["features"].tolist()

    while len(chars) > 0 and len(speech_toks) > 0:
        char_to_speech_ratio = len(chars) / len(speech_toks)

        if char_to_speech_ratio > 1:

