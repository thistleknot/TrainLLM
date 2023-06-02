#!/usr/bin/env python
# coding: utf-8

from common_imports import *
from functions import ConstantLengthDataset
from functions import estimate_chars_per_token, get_grouped_params, evaluate
from functions import create_dataset, create_dataloader, remove_extra_line_breaks
from functions import build_chain, traverse_chain
from functions import log_probs_from_logits, sequence_logprob, get_lr, filter_datasets_for_use_case, split_datasets, train_model, clear_model

#from dotenv import dotenv_values
sample = os.environ.get("sample")
sample = os.environ.get("sample_size")
with open('./resources/finetune_config.json', 'r') as f:
    finetune_config = json.load(f)

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sample = bool(os.environ.get("sample"))
sample_size = int(os.environ.get("sample_size"))

# Check if the script is being run directly
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=2000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--log_file_index", type=int, default=0)
    parser.add_argument("--enable_logging", action="store_true")

    cmd_args = parser.parse_args()

    finetune_config["learning_rate"] = cmd_args.learning_rate
    finetune_config["num_warmup_steps"] = cmd_args.num_warmup_steps
    finetune_config["gradient_accumulation_steps"] = cmd_args.gradient_accumulation_steps
    finetune_config["weight_decay"] = cmd_args.weight_decay
    finetune_config["max_norm"] = cmd_args.max_norm

    log_file = f'perplexity_logs_{cmd_args.log_file_index}.csv'
    log_headers = ['learning_rate', 'num_warmup_steps', 'gradient_accumulation_steps', 'weight_decay', 'max_norm', 'perplexity', 'eval_loss']

#load args
args = Namespace(**finetune_config)
# Reset random seed
set_seed(args.seed)

train_neo = False

if(train_neo):
    #load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt-neo-125M")
    # Load the saved model from the "./Alpha" directory
    model = AutoModelForCausalLM.from_pretrained("gpt-neo-125M")
else:
    tokenizer = AutoTokenizer.from_pretrained("gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("./Alpha")


with open('datasets_dict.pkl', 'rb') as f:
    datasets_dict = pickle.load(f)
finetune_datasets = filter_datasets_for_use_case(datasets_dict, 'finetune')
train_data_list, valid_data_list, valid_data_indices = split_datasets(finetune_datasets, ratio=0.7, random_state=args.seed)

train_data_list = [record for dataset in train_data_list.values() for record in dataset]
valid_data_list = [record for dataset in valid_data_list.values() for record in dataset]

print("finetune records:", len([*train_data_list, *valid_data_list]))

chars_per_token = estimate_chars_per_token(tokenizer, [*train_data_list, *valid_data_list])

#load args
args = Namespace(**finetune_config)
print(args)

# Reset random seed
set_seed(args.seed)

accelerator = Accelerator()
max_completed_steps = train_model(accelerator, model, tokenizer, train_data_list, valid_data_list, args, finetune_config, path="./Beta", chars_per_token=chars_per_token)

model = clear_model(accelerator, model, "./Beta")

accelerator = Accelerator()
train_model(accelerator, model, tokenizer, valid_data_list, valid_data_list, args, finetune_config, path="./Beta", chars_per_token=chars_per_token, skip_warmup=True, max_completed_steps=max_completed_steps)

