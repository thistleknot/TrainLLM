import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


tokenizer = AutoTokenizer.from_pretrained("gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("./Beta")
#model = AutoModelForCausalLM.from_pretrained("../bitsandbytes/bits")
app = FastAPI()


class QueryParameters(BaseModel):
    text: str
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 0
    top_p: Optional[float] = None
    do_sample: Optional[bool] = True
    max_length: int
    no_repeat_ngram_size: Optional[int] = 0
    num_beams: Optional[int] = 1

class OutputText(BaseModel):
    response: str


def generate_response_gpt_neo_raw(query: QueryParameters):
    print("Received params:", query)
    input_text = query.text

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)

    output = model.generate(
        input_ids,
        max_length=query.max_length,
        temperature=query.temperature,
        attention_mask=attention_mask,
        do_sample=query.do_sample,
        top_k=query.top_k,
        top_p=query.top_p,
        num_return_sequences=1,
        no_repeat_ngram_size=query.no_repeat_ngram_size,
        num_beams=query.num_beams,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


@app.post("/generate_response/gpt_neo", response_model=OutputText)
async def generate_response_gpt_neo(query: QueryParameters):
    response = generate_response_gpt_neo_raw(query)
    return {"response": response}


# Test runs before hosting the FastAPI server
common_params = {
    "text": "What is the meaning of life?",
    "max_length": 640,
    "do_sample": True,
}


beam_search_params = {
    **common_params,
    "num_beams": 5,
    "do_sample": False,
    "no_repeat_ngram_size": 2
}

top_k_sampling_params = {
    **common_params,
    "top_k": 50,
}

top_p_sampling_params = {
    **common_params,
    "top_p": 0.9,
}

greedy_decoding_params = {
    **common_params,
    "do_sample": False,
}

params_list = [
    ("beam", beam_search_params),
    ("top_k", top_k_sampling_params),
    ("top_p", top_p_sampling_params),
    ("greedy", greedy_decoding_params)
]

responses = {}
for name, params in params_list:
    query_params = QueryParameters(**params)
    response = generate_response_gpt_neo_raw(query_params)
    responses[name] = response
    print(f"{name}:\n{response}\n")
