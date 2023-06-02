# TrainLLM
download datasets (you will need to remove quotes, but if you want them, you can fine the file to do so in src)

run extract_quotes.py in src (optional), else comment out quotes dataset in data_prep.py
modify .env for the sample size you want

Because I use accelerate, you can use CPU or GPU, but the batch size is sized to 4GB of VRAM, so expect that, or modify the batch_size (if you have more VRAM)!

./run.sh (or run the files individually as outlined in run.sh)

curl -X POST "http://localhost:8000/generate-text" -H "Content-Type: application/json" -d '{"prompt": "What is the meaning of life?"}'

#ensure seed is set across both finetune and pretrain for train/test splits (i.e. it's currently 42) 

Dataset notes
n of 2550 with default ratio's will set values to their max without reducing any others due to max size limitations between datasets, this means on average 2550 records per dataset.
Training takes about 8 hours using a 4GB card, context length is set to 640 with strides of .5, so it's incrementing over 320 tokens at a time with a span of 640 tokens.

Dataset	MinSample
alpaca	3800
oa_conversation	3800
dolly15k	3800
supernatural	2372
squad_v2	2372
open_ai_summarize_tldr	2372
quotes	1913
cosmos_qa	1913
wiki_qa	1913
sciq	1913
subjqa	1913
squad	

Out of the above, sciq, squad_v2, and alpaca datasets have their context pretrained on so the context can be "generalized/memorized" by the model before asking the other split q/a sets of it.
However, for other contexts, they were trained during finetuning (in-context learning) because the information wasn't worthwhile to generalize on, but rather general english knowledge (semantics)

like subjqa

Default ratios are held here
(venv_train_neo) [root@pve0 venv_train_neo]# cat resources/data_prep.json
{
    "supernatural": {"ratio": 1.25},
    "squad_v2": {"ratio": 1.25},
    "openai_summarize_tldr": {"ratio": 1.25},
    "squad": {"ratio": "None"},
    "quotes": {"ratio": 1},
    "alpaca": {"ratio": 2},
    "cosmos_qa": {"ratio": 1},
    "wiki_qa": {"ratio": 1},
    "oa_conversation": {"ratio": 2},
    "dolly15k": {"ratio": 2},
    "sciq": {"ratio": 1},
    "subjqa": {"ratio": 1}
}

the formula is essentially 

dataset_sample_size = num_samples * ratio/np.nanmean(ratio)

and logic is in place to track the train test splits across pretrain and finetune (this was key), and 
ensuring that pretraining see's the context of these 3 required datasets during pretraining (i.e. 
during pretraining, finetune train_test split is performed to identify the context's of these 3 datasets 
to ensure pretraining continues on this dataset so when evaluation is done during finetuning, the context has already been seen).

pretraining and finetuning both continue training on the validation set using the same rules (but with no warmup_steps) that guided the first pass.  This is to ensure consistent pairing of the data as was done for actual training, however as there is no validation set, the validation set is simply used for both, but there are no warmups steps.  I find performance of epochs is similar to 1st pass (training on training data vs validation).

Note
    valid is used in pretrain_neo and finetune
    eval is used within train_llm of valid dataset


Generator settings are defined in
app.py
