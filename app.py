import gradio as gr
example_list =  [['examples/1.png'],
                 ['examples/2.png'],
                 ['examples/3.png'],
                 ['examples/4.png'],
                 ['examples/5.png'],
                 ['examples/6.png']]

title = "License plate reader 🚌"
description = "[Trained on european car plates] Identifies the license plate, cuts and displays it, and converts it into text. An image with higher resolution and clearer license plate will have a better accuracy."

gr.Interface.load("models/swww/test",examples=example_list,title=title,description=description).launch(debug=False,share=False)
