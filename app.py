# document summarization using huggingface transformers
from transformers import pipeline
import gradio as gr

# load model
model = pipeline('summarization')

# define function
def predict(prompt):
    summary = model(prompt)[0]["summary_text"]
    return summary

# launch interface
with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter text block to summarize", lines=4)
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

# launch interface
demo.launch()

