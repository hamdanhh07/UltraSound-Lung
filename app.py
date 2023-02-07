import gradio as gr
import requests
from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

dataset = load_dataset("hamdan07/UltraSound-lung")
image = Image.open(requests.get(dataset, stream=True).raw)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
inputs = feature_extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
API_URL = "https://api-inference.huggingface.co/models/hamdan07/UltraSound-Lung"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()
example_list =  [['examples/cov1.png'],
                 ['examples/cov2.jpg'],
                 ['examples/nor1.jpg'],
                 ['examples/nor2.jpg'],
                 ['examples/penu1.jpg'],
                 ['examples/penu2.jpg']]
title ="<p align='center'>COVID-19 Detection in Ultrasound with Timesformer</p>"
description ="<p style='text-align: center'> Trained on 500 data using Hugging Face dataset<br>It been traied using google/vit-base-patch16-224 </p><p style='text-align: center'>Link for the resource! <br> <a href='https://huggingface.co/datasets/hamdan07/UltraSound-lung' target='_blank'>Hugging Face Dataset</a> |<a href='https://huggingface.co/google/vit-base-patch16-224' target='_blank'>Model</a> | <a href='https://github.com/hamdanhh07/UltraSound-Lung' target='_blank'>github</a></p>"


   
    
   
gr.Interface.load("models/hamdan07/UltraSound-Lung",examples=example_list,title=title,description=description).launch(debug=False,share=False)

