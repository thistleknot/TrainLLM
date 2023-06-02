import torch
from transformers import AutoTokenizer, GPTNeoModel
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, Features, Array2D

import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def embed_text(examples):
    inputs = tokenizer(examples["text"], padding=True, truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model_output = model(**inputs)
    pooled_embeds = mean_pooling(model_output, inputs["attention_mask"])
    return {"embedding": pooled_embeds.cpu().numpy()}

model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-125m").to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer.pad_token = tokenizer.eos_token

wiki_dataset = concatenate_datasets([load_dataset("EleutherAI/wikitext_document_level",'wikitext-103-v1')[split] for split in ['train', 'validation', 'test']])
wiki_dataset = wiki_dataset.rename_column('page','text')

embeddings = wiki_dataset.map(embed_text, batched=True, batch_size=16)
embeddings.add_faiss_index("embedding")

index_path = "wiki_faiss_index.faiss"  # Specify the path where you want to save the index
embeddings.save_faiss_index("embedding", index_path)
index_path = "wiki_faiss_index.faiss"  # Specify the path where the index is saved
embeddings.load_faiss_index("embedding", index_path)

gc = pd.read_csv(f"H:\Curated-LLM-Data\data\graciousquotes.csv",index_col=0)
qa = pd.read_csv(f"H:\Curated-LLM-Data\data\quotes_by_author.csv",index_col=0)
quotes = pd.concat([gc,qa],axis=0).dropna().drop_duplicates()['0'].values

quotes_dataset = Dataset.from_pandas(pd.DataFrame({'text': quotes}))
quotes_embeddings = quotes_dataset.map(embed_text, batched=True, batch_size=4)
quotes_embeddings.add_faiss_index("embedding")

index_path = "quotes_faiss_index.faiss"  # Specify the path where you want to save the index
quotes_embeddings.save_faiss_index("embedding", index_path)
index_path = "quotes_faiss_index.faiss"  # Specify the path where the index is saved
quotes_embeddings.load_faiss_index("embedding", index_path)

query_text = 'What is the purpose of life?'

input_ids = tokenizer.encode(query_text, return_tensors="pt").to(model.device)
attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
with torch.no_grad():
    model_output = model(input_ids, attention_mask=attention_mask)
query_embedding = mean_pooling(model_output, attention_mask).cpu().numpy()

k = 3  # Select 3 nearest neighbors
scores, samples = quotes_embeddings.get_nearest_examples("embedding", query_embedding, k=k)

print("Context:")
responses = ''
for score, text in zip(scores, samples["text"]):
    #print("=" * 50,"\n")
    #print("Title:", text.split('\n')[0])
    #print(f"\nContent:\n\n{text}")
    #print(score, text)
    print(text)
    responses = responses+'\n'+text
    
print(f"Question: {query_text}")
print("Answer: ")
    

responses
