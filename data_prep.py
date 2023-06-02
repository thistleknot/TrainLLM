#!/usr/bin/env python
# coding: utf-8
from common_imports import *
from functions import ConstantLengthDataset
from functions import estimate_chars_per_token, get_grouped_params, evaluate
from functions import create_dataset, create_dataloader, remove_extra_line_breaks
from functions import build_chain, traverse_chain
from functions import log_probs_from_logits, sequence_logprob, get_lr, weighted_sample_sizes
from functions import generate_prompt_example

#from dotenv import dotenv_values
sample = os.environ.get("sample")
sample = os.environ.get("sample_size")
with open('./resources/pretrain_config.json', 'r') as f:
    pretrain_config = json.load(f)

with open('./resources/data_prep.json', 'r') as f:
    sample_ratios = json.load(f)
    sample_ratios = {k: {k2: (v2 if v2 != 'None' else None) for k2, v2 in v.items()} for k, v in sample_ratios.items()}

print(sample_ratios)

load_dotenv()

sample = bool(os.environ.get("sample"))
sample_size = int(os.environ.get("sample_size"))

pretrain_config['num_warmup_steps']=sample_size
pretrain_config['save_checkpoint_steps']=sample_size

torch.cuda.empty_cache()

#load args
args = Namespace(**pretrain_config)
# Reset random seed
set_seed(args.seed)

A=5e-5
B=3e-5
C=1e-5
C<B<A

#set to true if new model
if not os.path.exists("gpt-neo-125M/pytorch_model.bin"):
    new = True
else:
    new = False

model_name = 'EleutherAI/gpt-neo-125M'
model_dir = 'gpt-neo-125M'
print("models will be saved under models:", model_dir)

#these values have to be insync with their pretrain:finetune counterparts
#ratios might not work, because the sample size has to be equal between pretrain and finetune for a given subject.

#redpajama_dict = {'commoncrawl': .67, 'c4': .15, 'github': .045, 'wiki': .045, 'books': .045, 'arxiv': .025, 'stackexchange': .02}

selected_indices = []

# Load Dolly dataset
if os.path.exists("datasets.pkl"):
    with open("datasets.pkl", "rb") as f:
        #dolly15k, squad_v2, squad, sciq, alpaca, wiki_qa, cosmos_qa, supernatural, quotes, red_pajama_contexts, oa_dataset = pickle.load(f)
        dolly15k, squad_v2, squad, sciq, alpaca, wiki_qa, cosmos_qa, supernatural, quotes, oa_conversation_list, subjqa, openai_summarize_tldr = pickle.load(f)
else:
    dolly15k = concatenate_datasets([load_dataset("treadon/dolly-15k")[split] for split in ['train', 'validation']])
    squad_v2 = concatenate_datasets([load_dataset("squad_v2")[split] for split in ['train', 'validation']])
    squad = concatenate_datasets([load_dataset("squad")[split] for split in ['train', 'validation']])
    sciq = concatenate_datasets([load_dataset("sciq")[split] for split in ['train', 'validation','test']])
    alpaca = load_dataset("tatsu-lab/alpaca")['train']
    wiki_qa = concatenate_datasets([load_dataset("wiki_qa")[split] for split in ['train', 'validation','test']])
    subjqa = concatenate_datasets([load_dataset('subjqa', 'books')[split] for split in ['train', 'validation','test']])
    openai_summarize_tldr = concatenate_datasets([load_dataset("CarperAI/openai_summarize_tldr")[split] for split in ['train', 'valid', 'test']])

    """
    oa = load_dataset("h2oai/h2ogpt-oig-oasst1-instruct-cleaned-v2")
    oa_dataset = [
        {
            "human": text[text.find("<human>: ") + 8:text.find("<bot>:")].strip(),
            "bot": text[text.find("<bot>: ") + 6:].strip()
        }
        for text in oa['train']['input']
    ]
    """

    oa = concatenate_datasets([load_dataset("OpenAssistant/oasst1")[split] for split in ['train', 'validation']])
    # filter the dataset by lang == 'en'
    filtered_dataset = oa.filter(lambda example: example['lang'] == 'en')

    # Convert filtered_dataset to a DataFrame
    filtered_df = filtered_dataset.to_pandas()

    # Get the head_nodes
    head_nodes = filtered_df[filtered_df['parent_id'].isnull()]['message_id'].values

    # Initialize the conversation dictionary with a root key
    conversation_dict = {}

    # Initialize a list to collect end_of_chain_ids
    end_of_chain_ids = []

    # Iterate over the head_nodes
    for h in head_nodes:
        # Build the conversation chain for the current head_node
        chain, chain_end_of_chain_ids = build_chain(filtered_df, h)

        ## Add the chain to the conversation dictionary under the root key
        conversation_dict[h] = {
            'text': filtered_df.loc[filtered_df['message_id'] == h, 'text'].values[0],
            'children': chain
        }

        # Collect the end_of_chain_ids from the current chain
        end_of_chain_ids.extend(chain_end_of_chain_ids)

    # Display the conversation dictionary and end_of_chain_ids
    print(conversation_dict)
    print(end_of_chain_ids)

    conversations = []
    for head_message_id, head_content in conversation_dict.items():
        traverse_chain(head_content['children'], [head_content['text']], conversations)

    # Display the ragged array of conversations
    #print(conversations)
    [len(c) for c in conversations]

    # Initialize an empty list to store the context-response pairs
    context_response_pairs = []

    # Iterate over the conversations
    for conv in conversations:
        # Store the last element as 'Response' and the rest as 'Context'
        context_response_pairs.append({'Context': conv[:-1], 'Response': conv[-1]})

    # Convert the list of context-response pairs to a DataFrame
    conversation_df = pd.DataFrame(context_response_pairs)

    # Display the conversation DataFrame
    print(conversation_df.head())

    # Iterate over the conversation_df
    oa_conversation_list = []
    for i in conversation_df.iterrows():
        context = i[1]['Context']
        response = i[1]['Response']

        folded_context = ""

        if len(context) % 2 == 0:
            # Even case
            for j in range(0, len(context), 2):
                folded_context += f"Response: {context[j]}\nPrompt: {context[j+1]}\n"
        else:
            # Odd case
            for j in range(len(context)):
                label = "Prompt" if j % 2 == 0 else "Response"
                folded_context += f"{label}: {context[j]}\n"

        oa_conversation_list.append([folded_context,response])

    cosmos_qa = concatenate_datasets([load_dataset("cosmos_qa")[split] for split in ['train', 'validation','test']])
    supernatural = load_dataset("andersonbcdefg/supernatural-instructions-2m")['train']

    quotes_az = pd.read_csv(r"/mnt/distvol/text-generation-webui/data/quotes_by_author.csv",index_col=0)['0'].values
    quotes_gracious = pd.read_csv(r"/mnt/distvol/text-generation-webui/data/graciousquotes.csv",index_col=0)['0'].values
    quotes = [*quotes_az,*quotes_gracious]

    with open("datasets.pkl", "wb") as f:
        #pickle.dump((dolly15k, squad_v2, squad, sciq, alpaca, wiki_qa, cosmos_qa, supernatural, quotes, red_pajama_contexts, oa_dataset), f)
        pickle.dump((dolly15k, squad_v2, squad, sciq, alpaca, wiki_qa, cosmos_qa, supernatural, quotes, oa_conversation_list, subjqa, openai_summarize_tldr), f)

if sample:
    #pretrain
    #finetune

    # Sample indices for common datasets (e.g., Dolly, SQuAD v2, SCI-Q)
    
    sample_ratios['dolly15k']['size'] = len(dolly15k)
    sample_ratios['squad_v2']['size'] = len(squad_v2)
    sample_ratios['squad']['size'] = len(squad)
    sample_ratios['sciq']['size'] = len(sciq)
    sample_ratios['alpaca']['size'] = len(alpaca['text'])
    sample_ratios['wiki_qa']['size'] = len(wiki_qa)
    sample_ratios['cosmos_qa']['size'] = len(cosmos_qa)
    sample_ratios['supernatural']['size'] = len(supernatural)
    sample_ratios['openai_summarize_tldr']['size'] = len(openai_summarize_tldr)
    sample_ratios['oa_conversation']['size'] = len(oa_conversation_list)
    sample_ratios['subjqa']['size'] = len(subjqa)
    sample_ratios['quotes']['size'] = len(quotes)

    df_ratios = pd.DataFrame(sample_ratios).T
    df_ratios['ratios'] = np.round(df_ratios['ratio']/np.nanmean(df_ratios['ratio']),2)
    
    df_ratios['sample_size'] = df_ratios['ratios'] * sample_size
    df_ratios['min_sample_size'] = np.minimum(df_ratios['size'], df_ratios['sample_size']).where(df_ratios['sample_size'].notna(), np.nan)
    
    for v in sample_ratios.values():
        ratio = v['ratio']
        if ratio is None:
            v['ratio_new'] = None
            v['sample_size'] = None
            v['min_sample_size'] = None
        else:
            mean_ratio = np.nanmean(df_ratios['ratios'])
            ratio_new = np.round(ratio / mean_ratio,2)
            v['ratio_new'] = ratio_new
            v['sample_size'] = ratio_new * sample_size
            v['min_sample_size'] = np.minimum(v['size'], v['sample_size'])

    print(sample_ratios)
    print(pd.DataFrame(sample_ratios).T)

    dolly15k_indices = random.sample(range(sample_ratios['dolly15k']['size']), int(sample_ratios['dolly15k']['min_sample_size']))
    sample_ratios['dolly15k']['indices']=dolly15k_indices
    dolly15k_context_records = [dolly15k[i]['context'] for i in dolly15k_indices]
    dolly15k_qa_records = [generate_prompt_example(**{'prompt': dolly15k[i]['instruction'], 'response': dolly15k[i]['response']}) for i in dolly15k_indices]

    squad_v2_indices = random.sample(range(sample_ratios['squad_v2']['size']), int(sample_ratios['squad_v2']['min_sample_size']))
    sample_ratios['squad_v2']['indices']=dolly15k_indices
    squad_v2_context_records = [squad_v2[i]['context'] for i in squad_v2_indices]
    squad_v2_qa_records = [generate_prompt_example(**{'prompt': squad_v2[i]['question'], 'response': squad_v2[i]['answers']['text']}) for i in squad_v2_indices]

    """
    squad_indices = random.sample(range(sample_ratios['squad']['size']), int(sample_ratios['squad']['min_sample_size']))
    sample_ratios['squad']['indices']=dolly15k_indices
    squad_context_records = [squad[i]['context'] for i in squad_indices]
    squad_qa_records = [generate_prompt_example(**{'prompt': squad[i]['question'], 'response': squad[i]['answers']['text']}) for i in squad_indices]
    """
    sciq_indices = random.sample(range(sample_ratios['sciq']['size']), int(sample_ratios['sciq']['min_sample_size']))
    sample_ratios['sciq']['indices']=dolly15k_indices
    sciq_context_records = [sciq[i]['support'] for i in sciq_indices]
    sciq_records = [generate_prompt_example(**{'prompt': sciq[i]['question'], 'response': sciq[i]['correct_answer']}) for i in sciq_indices]

    alpaca_indices = random.sample(range(sample_ratios['alpaca']['size']), int(sample_ratios['alpaca']['min_sample_size']))
    sample_ratios['alpaca']['indices']=dolly15k_indices
    alpaca_qa_records = [generate_prompt_example(**{'prompt': alpaca[i]['instruction'], 'response': alpaca[i]['output']}) for i in alpaca_indices]

    wiki_qa_indices = random.sample(range(sample_ratios['wiki_qa']['size']), int(sample_ratios['wiki_qa']['min_sample_size']))
    sample_ratios['wiki_qa']['indices']=dolly15k_indices
    wiki_qa_records = [generate_prompt_example(**{'prompt': wiki_qa[i]['question'], 'response': wiki_qa[i]['answer']}) for i in wiki_qa_indices]

    cosmos_qa_indices = random.sample(range(sample_ratios['cosmos_qa']['size']), int(sample_ratios['cosmos_qa']['min_sample_size']))
    sample_ratios['cosmos_qa']['indices']=dolly15k_indices
    cosmos_qa_records = [generate_prompt_example(**{'context': cosmos_qa[i]['context'], 'prompt': cosmos_qa[i]['question'], 'response': cosmos_qa[i][f'answer{cosmos_qa[i]["label"]}'] if cosmos_qa[i]['label'] != -1 else ''}) for i in cosmos_qa_indices]

    supernatural_indices = random.sample(range(sample_ratios['supernatural']['size']), int(sample_ratios['supernatural']['min_sample_size']))
    sample_ratios['supernatural']['indices']=dolly15k_indices
    supernatural_records = [generate_prompt_example(**{'prompt': supernatural[i]['prompt'], 'response': supernatural[i]['response']}) for i in supernatural_indices]
    
    openai_summarize_tldr_indices = random.sample(range(sample_ratios['openai_summarize_tldr']['size']), int(sample_ratios['openai_summarize_tldr']['min_sample_size']))
    sample_ratios['openai_summarize_tldr']['indices']=dolly15k_indices
    openai_summarize_tldr_records = [generate_prompt_example(**{'context': openai_summarize_tldr[i]['prompt'], 'prompt': 'TL;DR', 'response': openai_summarize_tldr[i]['label']}) for i in openai_summarize_tldr_indices]

    oa_indices = random.sample(range(sample_ratios['oa_conversation']['size']), int(sample_ratios['oa_conversation']['min_sample_size']))
    sample_ratios['oa_conversation']['indices']=dolly15k_indices
    oa_qa_records = [generate_prompt_example(**{'prompt': oa_conversation_list[i][0], 'response': oa_conversation_list[i][1]}, cot=True) for i in oa_indices]

    #oa_finetune_records = [generate_prompt_example(**{'prompt': oa_dataset[i]['human'], 'response': oa_dataset[i]['bot']}) for i in oa_indices]
    #oa_finetune_records = [generate_prompt_example(**{'prompt': c[0], 'response': c[1]}, cot=True) for c in oa_conversation_list]

    subjqa_indices = random.sample(range(sample_ratios['subjqa']['size']), int(sample_ratios['subjqa']['min_sample_size']))
    sample_ratios['subjqa']['indices']=dolly15k_indices
    subjqa_qa_records = [generate_prompt_example(**{'context': subjqa[i]['context'], 'prompt': subjqa[i]['question'], 'response': subjqa[i]['answers']['text']}) for i in subjqa_indices]

    if(False):
        red_pajamas_indices = random.sample(range(len(red_pajama_contexts)), int(sample_ratios['red_pajamas']['min_sample_size']))
        #red_pajamas_records = [red_pajama_contexts[i] for i in red_pajamas_indices]

    quotes_indices = random.sample(range(sample_ratios['quotes']['size']), int(sample_ratios['quotes']['min_sample_size']))
    sample_ratios['quotes']['indices']=dolly15k_indices
    quotes_context_records = [quotes[i] for i in quotes_indices]

datasets_dict = {
    'dolly15k': {'pretrain': dolly15k_context_records, 'finetune': dolly15k_qa_records},
    'squad_v2': {'pretrain': squad_v2_context_records, 'finetune': squad_v2_qa_records},
    'sciq': {'pretrain': sciq_context_records, 'finetune': sciq_records},
    'quotes': {'pretrain': quotes_context_records, 'finetune': None},
    'wiki_qa': {'pretrain': None, 'finetune': wiki_qa_records},
    'alpaca': {'pretrain': None, 'finetune': alpaca_qa_records},
    'cosmos_qa': {'pretrain': None, 'finetune': cosmos_qa_records},
    'supernatural': {'pretrain': None, 'finetune': supernatural_records},
    'oa_conversation': {'pretrain': None, 'finetune': oa_qa_records},
    'subjqa_qa': {'pretrain': None, 'finetune': subjqa_qa_records},
    'openai_summarize_tldr': {'pretrain': None, 'finetune': openai_summarize_tldr_records},
}

#pretrain_records_sample = [record for record in [*dolly15k_context_records, *squad_v2_context_records, *sciq_context_records, *quotes_context_records] if record and not isinstance(record, float)]
#finetune_records_sample = [*dolly15k_qa_records,*squad_v2_qa_records,*sciq_records,*wiki_qa_records,*alpaca_qa_records,*cosmos_qa_records,*supernatural_records, *oa_qa_records, *subjqa_qa_records, *openai_summarize_tldr_records]

#pickle.dump(pretrain_records_sample, open('pretrain_records_sample.pkl', 'wb'))
#pickle.dump(finetune_records_sample, open('finetune_records_sample.pkl', 'wb'))
for dataset_name, use_cases in datasets_dict.items():
    for use_case_name, records in use_cases.items():
        if records:
            print(f"Size of {dataset_name}:{use_case_name} records = {len(records)}")
        else:
            print(f"No records for {dataset_name}:{use_case_name}")

pickle.dump(datasets_dict, open('datasets_dict.pkl', 'wb'))
#with open('./data/datasets_dict.json', 'w') as f:
#    json.dump(datasets_dict, f) 
