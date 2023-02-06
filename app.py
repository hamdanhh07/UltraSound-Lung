import gradio as gr
import requests

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
title ="# **<p align='center'>COVID-19 Detection in Ultrasound with Timesformer</p>**"
description ="""
        <p style='text-align: center'>
        Trained on 500 data using Hugging Face dataset..
        <br>It is released by Facebook AI Research in ICML 2021.
        <br>      
        <p style='text-align: center'>
        Follow me for more! 
        <br> <a href='https://twitter.com/fcakyon' target='_blank'>twitter</a> | <a href='https://github.com/fcakyon' target='_blank'>github</a> | <a href='https://www.linkedin.com/in/fcakyon/' target='_blank'>linkedin</a> | <a href='https://fcakyon.medium.com/' target='_blank'>medium</a>
        </p>
        .
        </p>
        """


   
    
   
gr.Interface.load("models/hamdan07/UltraSound-Lung",examples=example_list,title=title,description=description).launch(debug=False,share=False)

