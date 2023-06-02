# main.py
import torch
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Use CUDA if available
if torch.cuda.is_available():
    model = model.to("cuda")

# Pydantic model for input validation
class Query(BaseModel):
    text: str
    temperature: float
    top_k: int
    top_p: float
    max_length: int

@app.post("/summarize")
async def summarize(query: Query):
    # Generate TL;DR summary by prompting GPT-2
    input_ids = tokenizer.encode("TL;DR: " + query.text, return_tensors="pt")
    
    # Move input to GPU if available
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    
    output = model.generate(
        input_ids,
        max_length=query.max_length,
        temperature=query.temperature,
        top_k=query.top_k,
        top_p=query.top_p,
        num_return_sequences=1,
    )

    # Decode the generated text and return it
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"summary": summary}

# Gradio function for summarization
def gradio_summarize(text: str, temperature: float, top_k: int, top_p: float, max_length: int):
    query = Query(text=text, temperature=temperature, top_k=top_k, top_p=top_p, max_length=max_length)
    result = summarize(query)
    return result

# Gradio interface
iface = gr.Interface(
    gradio_summarize,
    inputs=[
        gr.inputs.Textbox(label="Text"),
        gr.inputs.Slider(0.1, 2.0, step=0.1, default=1.0, label="Temperature"),
        gr.inputs.Number(default=50, label="Top K"),
        gr.inputs.Slider(0.0, 1.0, step=0.01, default=1.0, label="Top P"),
        gr.inputs.Number(default=2048, label="Max Length"),
    ],
    outputs=gr.outputs.Textbox(label="Summary"),
)

# Launch Gradio app
iface.launch()
