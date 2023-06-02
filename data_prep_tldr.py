# Load data
from common_imports import *
from functions import ConstantLengthDataset, TextGenerationInput
from functions import estimate_chars_per_token, get_grouped_params, evaluate
from functions import create_dataset, create_dataloader, remove_extra_line_breaks
from functions import generate_prompt_example, build_chain, traverse_chain
from functions import log_probs_from_logits, sequence_logprob, get_lr, generate_text, estimate_chars_per_token


#from dotenv import dotenv_values
sample = os.environ.get("sample")
sample = os.environ.get("sample_size")
with open('./resources/pretrain_config.json', 'r') as f:
    pretrain_config = json.load(f)

load_dotenv()

sample = bool(os.environ.get("sample"))
sample_size = int(os.environ.get("sample_size"))

torch.cuda.empty_cache()

#load args
args = Namespace(**pretrain_config)
# Reset random seed
set_seed(args.seed)

with open('./resources/pretrain_config.json', 'r') as f:
    pretrain_config = json.load(f)

#set to true if new model
if not os.path.exists("gpt-neo-125M/pytorch_model.bin"):
    new = True
else:
    new = False

if(new):
    # Load and tokenize the smaller GPT-Neo model (125M parameters)
    model_name = 'EleutherAI/gpt-neo-125M'
    model_dir = 'gpt-neo-125M'
    print("models will be saved under models:",model_dir)
    # check if model already exists
    #if not os.path.exists(model_path):
    if(True):
        # download model
        print("loading model from hf")
        model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m",gradient_checkpointing=True)#.to(device)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m",gradient_checkpointing=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        # create directory if it doesn't exist
    if(False):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            # save model to disk
            model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m",gradient_checkpointing=True)#.to(device)
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m",gradient_checkpointing=True)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        else:
            print("loading model from disk")
            # load model from disk
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = GPTNeoForCausalLM.from_pretrained(model_dir,gradient_checkpointing=True)#.to(device)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
else:
    # Load the saved model checkpoint
    print("loading prior downloaded model")
    model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m')
    tokenizer = GPT2TokenizerFast.from_pretrained('EleutherAI/gpt-neo-125m')

pretrain_records_sample = pickle.load(open('pretrain_records_sample.pkl', 'rb'))
#finetune_records_sample = pickle.load(open('finetune_records_sample.pkl', 'rb'))
print("after load")

chars_per_token = estimate_chars_per_token(tokenizer, pretrain_records_sample)

stride_dataset = create_dataset(tokenizer, pretrain_records_sample, chars_per_token)

# Create an iterator for the stride_dataset
stride_dataset_iterator = iter(stride_dataset)

# Initialize an empty list to store decoded text between eos tokens
decoded_texts = [[]]
current_tokens = []
eos_token_id = tokenizer.eos_token_id

# Iterate over the stride_dataset_iterator
for stride in stride_dataset_iterator:
    tokens = stride.tolist()

    # Find all eos token indices
    eos_indices = [i for i, token in enumerate(tokens) if token == eos_token_id]

    # If eos token is found, split the tokens accordingly
    if eos_indices:
        for i, eos_index in enumerate(eos_indices):
            if i == 0:
                # Take tokens from the beginning of the stride to the first eos token
                segment = tokens[:eos_index]
            else:
                # Take tokens between successive eos tokens
                segment = tokens[eos_indices[i - 1] + 1 : eos_index]

            current_tokens.extend(segment)
            decoded_text = tokenizer.decode(current_tokens)
            decoded_texts[-1].append(decoded_text)
            current_tokens = []

            # Start a new list for the next set of decoded tokens
            decoded_texts.append([])
        
        # Add remaining tokens after the last eos token to current_tokens
        current_tokens.extend(tokens[eos_indices[-1] + 1:])
    else:
        # If no eos token is found, just extend the current_tokens
        current_tokens.extend(tokens)

    # Add the remaining tokens if the iterator ends without an eos token
if current_tokens:
    decoded_text = tokenizer.decode(current_tokens)
    decoded_texts[-1].append(decoded_text)

# Apply TL;DR on the grouped decoded_texts and store them in a similar nested list structure
tldrs = []
for stride_group in decoded_texts:
    stride_tldrs = []
    for decoded_text in stride_group:
        data = {
            "prompt": "Please provide a TL;DR summary for the following text:\n" + decoded_text + "\nResponse:"
        }
        response = requests.post(url, headers=headers, json=data)
        tldr = json.loads(response.text)['models_output'][0]['generated_text']
        stride_tldrs.append(tldr)
    tldrs.append(stride_tldrs)

# Combine the TL;DR summaries into a continuous sequence of characters for each stride group
tldr_sequences = ["\n".join(stride_tldrs) for stride_tldrs in tldrs]

# Apply a final TL;DR for each tldr_sequence with the given prompt
final_tldrs = []

for i, tldr_sequence in enumerate(tldr_sequences):
    prompt = ("These are TL;DR's of stride based context lengths, 50% crossover of repeated information exists "
              "between preceding and next record (except the first record has no prior, and the last record has no next). "
              "Synthesize the above TL;DR with this understanding:\n" + tldr_sequence + "\nTL;DR")

    data = {"prompt": prompt}
    response = requests.post(url, headers=headers, json=data)
    final_tldr = json.loads(response.text)['models_output'][0]['generated_text']
    final_tldrs.append(final_tldr)

# Print the final TL;DR summaries
for i, final_tldr in enumerate(final_tldrs):
    print(f"Final TL;DR {i + 1}:")
    print(final_tldr)
    print()

# Create a new ConstantLengthDataset instance with the TL;DR tokens

pickle.dump(final_tldr, open('new_stride_dataset.pkl', 'wb'))

chars_per_token = estimate_chars_per_token(tokenizer, final_tldr)

tldr_dataset = create_dataset(tokenizer, final_tldr, chars_per_token)

