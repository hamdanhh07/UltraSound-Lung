from transformers import AutoFeatureExtractor, AutoModelForImageClassification
extractor = AutoFeatureExtractor.from_pretrained("hamdan07/UltraSound-Lung")
model = AutoModelForImageClassification.from_pretrained("hamdan07/UltraSound-Lung")
