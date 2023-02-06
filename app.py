import gradio as gr
example_list =  [['examples/cov1.png'],
                 ['examples/cov2.jpg'],
                 ['examples/nor1.png'],
                 ['examples/nor2.jpg'],
                 ['examples/penu1.jpg'],
                 ['examples/penu2.jpg']]
from huggingface_hub.inference_api import InferenceApi

inference = InferenceApi(repo_id="hamdan07/UltraSound-Lung", token="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")

title = "COVID-19 Detection in Ultrasound Imagery Using Artificial intelligent Methods"
description = "[Trained on european car plates] Identifies the license plate, cuts and displays it, and converts it into text. An image with higher resolution and clearer license plate will have a better accuracy."

gr.Interface.load("inference",examples=example_list,title=title,description=description).launch(debug=False,share=False)
