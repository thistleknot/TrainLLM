from common_imports import *

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from functions import ConstantLengthDataset, TextGenerationInput
from functions import estimate_chars_per_token, get_grouped_params, evaluate
from functions import create_dataset, create_dataloader, remove_extra_line_breaks
from functions import generate_prompt_example, build_chain, traverse_chain
#from functions import log_probs_from_logits, sequence_logprob, get_lr, generate_text
#from functions import generate_text_endpoint, TextGenerationInput

import torch


tokenizer = AutoTokenizer.from_pretrained("gpt-neo-125M")

accelerator = Accelerator()

model = AutoModelForCausalLM.from_pretrained("./Beta")
model = accelerator.prepare(model)

app = FastAPI()

class QueryParameters(BaseModel):
    text: str
    temperature: float
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_length: int
    no_repeat_ngram_size: int
    num_beams: Optional[int] = 1

class OutputText(BaseModel):
    response: str

@app.post("/generate_response/gpt_neo", response_model=OutputText)
async def generate_response_gpt_neo(query: QueryParameters):
    print("Received params:", query)  # Debug statement to check received parameters
    input_text = query.text

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    input_ids = accelerator.prepare(input_ids)
    attention_mask = accelerator.prepare(attention_mask)

    # Move input tensors to the same device as the model
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    output = model.generate(
        input_ids,
        max_length=query.max_length,
        temperature=query.temperature,
        attention_mask=attention_mask,
        top_k=query.top_k,
        top_p=query.top_p,
        num_return_sequences=1,
        no_repeat_ngram_size=query.no_repeat_ngram_size,
        num_beams=query.num_beams,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    response = generated_text

    return {"response": response}
