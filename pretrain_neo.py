
#!/usr/bin/env python
# coding: utf-8

from common_imports import *
from functions import ConstantLengthDataset
from functions import estimate_chars_per_token, get_grouped_params, evaluate
from functions import create_dataset, create_dataloader, remove_extra_line_breaks
from functions import build_chain, traverse_chain
from functions import log_probs_from_logits, sequence_logprob, get_lr, filter_datasets_for_use_case, split_datasets, train_model, clear_model
#from functions import parse_args_and_update_config, log_to_csv

def log_to_csv(log_file, data):
    with open(log_file, mode='a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

#load config
with open('./resources/pretrain_config.json', 'r') as f:
    pretrain_config = json.load(f)
    
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sample = bool(os.environ.get("sample"))
sample_size = int(os.environ.get("sample_size"))

# pretrain_neo.py

# Check if the script is being run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    #parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--log_file_index", type=int, default=0)
    parser.add_argument("--enable_logging", action="store_true")

    cmd_args = parser.parse_args()

    pretrain_config["learning_rate"] = cmd_args.learning_rate
    #pretrain_config["num_warmup_steps"] = cmd_args.num_warmup_steps
    pretrain_config["gradient_accumulation_steps"] = cmd_args.gradient_accumulation_steps
    pretrain_config["weight_decay"] = cmd_args.weight_decay
    pretrain_config["max_norm"] = cmd_args.max_norm

    log_file = f'perplexity_logs_{cmd_args.log_file_index}.csv'
    log_headers = ['learning_rate', 'num_warmup_steps', 'gradient_accumulation_steps', 'weight_decay', 'max_norm', 'perplexity', 'eval_loss']

# Your existing pretrain_neo.py code here
# Use the updated pretrain_config values

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

print("before load")
with open('datasets_dict.pkl', 'rb') as f:
    datasets_dict = pickle.load(f)
pretrain_datasets = filter_datasets_for_use_case(datasets_dict, 'pretrain')
finetune_datasets = filter_datasets_for_use_case(datasets_dict, 'finetune')

train_data_list, valid_data_list, valid_data_indices = split_datasets(finetune_datasets, ratio=0.7, random_state=args.seed)

#ftrain_data_list, fvalid_data_list = split_datasets(finetune_datasets, ratio=0.7, random_state=42)

# Filter out blank or np.nan values for each dataset in pretrain_train_data
filtered_train_data_list = {}
for key, value in train_data_list.items():
    filtered_train_data_list[key] = [record for record in value if record and not isinstance(record, float)]

# Do the same for pretrain_test_data if needed
filtered_valid_data_list = {}
for key, value in valid_data_list.items():
    filtered_valid_data_list[key] = [record for record in value if record and not isinstance(record, float)]

filtered_train_data_list = [record for dataset in filtered_train_data_list.values() for record in dataset]
filtered_valid_data_list = [record for dataset in filtered_valid_data_list.values() for record in dataset]

#add back in context related data that will be validated on during finetuning
post_train_datasets = ['sciq', 'squad_v2', 'dolly15k']
post_train_data_list = []

#train finetune evaluation context records during pretraining
for dataset in post_train_datasets:
    pretrain_data = datasets_dict[dataset]['pretrain']
    valid_indices = valid_data_indices[dataset]

    for index in valid_indices:
        record = pretrain_data[index]
        if record and not isinstance(record, float):
            post_train_data_list.append(record)

#filter out records already captured during training and evaluation periods so I don't double dip
post_train_data_list = [record for record in post_train_data_list if record not in filtered_train_data_list]
post_train_data_list = [record for record in post_train_data_list if record not in filtered_valid_data_list]
#filtered_train_data_list += additional_pretrain_records

#doesn't need to be explicit.  Data is assumed generalized similar to other unstructured data.
#post_train_data_list = ['Context: ' + record for record in post_train_data_list]

chars_per_token = estimate_chars_per_token(tokenizer, [*filtered_train_data_list,*filtered_valid_data_list, *post_train_data_list])
print("after chars_per_token")

accelerator = Accelerator()
max_completed_steps = train_model(accelerator, model, tokenizer, filtered_train_data_list, filtered_valid_data_list, args, pretrain_config, path="./Alpha", chars_per_token=chars_per_token)
model = clear_model(accelerator, model, "./Alpha")

accelerator = Accelerator()
max_completed_steps = train_model(accelerator, model, tokenizer, filtered_valid_data_list, filtered_valid_data_list, args, pretrain_config, path="./Alpha", chars_per_token=chars_per_token, skip_warmup=True, max_completed_steps=max_completed_steps)
model = clear_model(accelerator, model, "./Alpha")

accelerator = Accelerator()
model = train_model(accelerator, model, tokenizer, post_train_data_list, post_train_data_list, args, pretrain_config, path="./Alpha", chars_per_token=chars_per_token, skip_warmup=True, max_completed_steps=max_completed_steps)
