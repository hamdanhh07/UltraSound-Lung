import gradio as gr
example_list =  [['examples/cov1.png'],
                 ['examples/cov2.jpg'],
                 ['examples/nor1.png'],
                 ['examples/nor2.jpg'],
                 ['examples/penu1.jpg'],
                 ['examples/penu2.jpg']]
import requests

API_URL = "https://api-inference.huggingface.co/models/hamdan07/UltraSound-Lung"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

title = "COVID-19 Detection in Ultrasound Imagery Using Artificial intelligent Methods"
description = "[Trained on european car plates] Identifies the license plate, cuts and displays it, and converts it into text. An image with higher resolution and clearer license plate will have a better accuracy."

gr.Interface.load("models/hamdan07/UltraSound-Lung"),examples=example_list,title=title,description=description).launch(debug=False,share=False)
