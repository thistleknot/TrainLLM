#!/usr/bin/env python
# coding: utf-8

from common_imports import *
from torch.nn.utils import clip_grad_norm_

#from dotenv import dotenv_values
sample = os.environ.get("sample")
sample = os.environ.get("sample_size")
with open('./resources/pretrain_config.json', 'r') as f:
    pretrain_config = json.load(f)

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

"""
class ConstantLengthDataset(IterableDataset):

    def __init__(self, tokenizer, dataset, seq_length=args.seq_length,
                 num_of_sequences=args.batch_size, chars_per_token=3.6, eos=''):
        self.tokenizer = tokenizer
        self.concat_token_id = eos
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m=f"Buffer full: {buffer_len}>={self.input_characters:.0f}"
                    #print(m)
                    break
                try:
                    m=f"Fill buffer: {buffer_len}<{self.input_characters:.0f}"
                    #print(m)
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for i, tokenized_input in enumerate(tokenized_inputs['input_ids']):
                if not all(isinstance(x, int) for x in tokenized_input):
                    print(f"Problematic tokenized input: {tokenized_input}")
                    print(f"Original string: {buffer[i]}")                        
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
"""
class ConstantLengthDataset(IterableDataset):

    def __init__(self, tokenizer, dataset, seq_length=args.seq_length,
                 num_of_sequences=args.batch_size, chars_per_token=3.6, eos='', stride_ratio=0.5):
        self.tokenizer = tokenizer
        self.concat_token_id = eos
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences
        self.stride_ratio = stride_ratio

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)["content"])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for i, tokenized_input in enumerate(tokenized_inputs['input_ids']):
                if not all(isinstance(x, int) for x in tokenized_input):
                    print(f"Problematic tokenized input: {tokenized_input}")
                    print(f"Original string: {buffer[i]}")
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            # Apply sliding window approach
            stride_length = int(self.seq_length * self.stride_ratio)
            for i in range(0, len(all_token_ids), stride_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)

def estimate_chars_per_token(tokenizer, records_):
    num_samples = len(records_)
    total_characters = 0
    total_tokens = 0

    # Use random.sample to get a subset of records
    #sampled_records = random.sample(records_, num_samples)
    sampled_records = records_

    for record in sampled_records:
        total_characters += len(record)
        total_tokens += len(tokenizer.tokenize(record))

    return total_characters / total_tokens

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [{'params': params_with_wd, 'weight_decay': args.weight_decay},
            {'params': params_without_wd, 'weight_decay': 0.0}]

def evaluate(model, eval_dataloader, accelerator):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        #batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps: break
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()

def create_dataset(tokenizer, data_list, chars_per_token):
    # Create the datasets with the required format
    data = datasets.Dataset.from_dict({"content": data_list})
    dataset = ConstantLengthDataset(tokenizer, data, seq_length=args.seq_length,num_of_sequences=args.batch_size, chars_per_token=chars_per_token,eos=tokenizer.eos_token_id)
    return dataset

def create_dataloader(dataset_name):
    dataloader=DataLoader(dataset_name, batch_size=args.batch_size)
    #prepared_dataloader = accelerator.prepare(dataloader)
    return dataloader

def remove_extra_line_breaks(text):
    cleaned_text = re.sub(r'\n\n', '', text).strip()
    return cleaned_text

def generate_prompt_example(cot=False, **kwargs):
    context = kwargs.get('context', '')
    prompt = kwargs['prompt']
    response = kwargs['response']

    # Check if response is a list and join elements if it is
    response_str = ', '.join(response) if isinstance(response, list) else response

    if(context==''):
        context = ''
    else:
        context = 'Context:\n' + remove_extra_line_breaks(context) + '\n'
    if(cot):
        string = context + f"Prompt:\n{remove_extra_line_breaks(prompt)}\nResponse:\n{remove_extra_line_breaks(response_str)}"
    else:
        string = context + f"Prompt:\n{remove_extra_line_breaks(prompt)}\nResponse:\n{remove_extra_line_breaks(response_str)}"

    return string

def build_chain(chain_df, parent_id):
    chain = {}
    end_of_chain_ids = []

    # Get the rows with the given parent_id
    child_rows = chain_df[chain_df['parent_id']==parent_id]

    if child_rows.empty:
        end_of_chain_ids.append(parent_id)
    else:
        # Iterate over the child rows
        for _, row in child_rows.iterrows():
            # Recursively call the function with the current message_id
            child_chain, child_end_of_chain_ids = build_chain(chain_df, row['message_id'])

            # Add the current message_id, text, and child_chain to the chain dictionary
            chain[row['message_id']] = {
                'text': row['text'],
                'children': child_chain
            }

            # Collect the end_of_chain_ids from the child_chain
            end_of_chain_ids.extend(child_end_of_chain_ids)

    return chain, end_of_chain_ids

def traverse_chain(chain, conversation, conversations):
    if not chain:
        # Base case: end of the chain
        conversations.append(conversation)
    else:
        for message_id, content in chain.items():
            # Recursively call the function for each child chain
            traverse_chain(content['children'], conversation + [content['text']], conversations)

def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label

def sequence_logprob(model, accelerator, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(
            output.logits[:, :-1, :], labels[:, 1:])
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return accelerator.gather(seq_log_prob).cpu().numpy()

def get_lr():
    return optimizer.param_groups[0]['lr']

def log_to_csv(log_file, data):
    with open(log_file, mode='a') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def save_hyperparameter_combinations(hyperparameter_combinations, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write the header row
        writer.writerow(["log_file_index", "learning_rate", "num_warmup_steps", "gradient_accumulation_steps", "weight_decay", "max_norm"])
        # Write the hyperparameter combinations
        for index, combination in enumerate(hyperparameter_combinations):
            writer.writerow([index] + list(combination))

def weighted_sample_sizes(total_sample_size, weights):
    sample_sizes = {}
    total_weights = sum(weights.values())
    for dataset, weight in weights.items():
        sample_sizes[dataset] = int(total_sample_size * weight / total_weights)

    return sample_sizes

def filter_datasets_for_use_case(datasets, use_case):
    filtered_datasets = {}
    for key, value in datasets.items():
        if value[use_case]:
            filtered_datasets[key] = value[use_case]
    return filtered_datasets

def split_datasets(data_dict, ratio=0.7, random_state=None):
    train_data = {}
    test_data = {}
    validation_indices = {}
    
    for key, value in data_dict.items():
        train, test, train_indices, test_indices = train_test_split(value, range(len(value)), train_size=ratio, random_state=random_state)
        train_data[key] = train
        test_data[key] = test
        validation_indices[key] = test_indices
        
    return train_data, test_data, validation_indices

def unique_elements(lst):
    result = []
    seen = set()
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

#train_model(accelerator, model, tokenizer, post_train_data_list, filtered_valid_data_list, args, pretrain_config, path="./Alpha", skip_warmup=True)
def train_model(accelerator, model, tokenizer, train_data_list_to_use, eval_data_list_to_use, args, config, path, chars_per_token, skip_warmup=False, max_completed_steps=None):

    # Initialize early stopping variables
    best_model_state = None
    best_perplexity = np.inf
    max_consecutive_failures = 3
    stored_models = []

    completed_steps = 0
    desired_perplexity = 0.05

    consecutive_failures = 0

    print("train records:", len(unique_elements([*train_data_list_to_use,*eval_data_list_to_use])))
    total_tokens = sum([len(tokenizer.tokenize(record)) for record in train_data_list_to_use])
    stride_length = int(args.seq_length * args.stride_ratio)
    # Calculate the number of tokens processed per batch
    tokens_per_batch = args.seq_length * args.batch_size

    # Calculate the number of training steps needed to iterate over all samples
    max_train_steps = (total_tokens - args.seq_length) // stride_length
    max_train_steps = max_train_steps // args.batch_size

    # Divide max_train_steps by gradient_accumulation_steps
    max_train_steps = int(np.max([2, max_train_steps // config['gradient_accumulation_steps']]))

    print(f"Total tokens: {total_tokens}")
    print(f"Number of training steps: {max_train_steps}")

    #3 epochs (no more than, logic is in place to stop training after 1/3 of this value if perplexity doesn't improve 3 consecutive times in a row.
    #Note: Model improvement quality is affected by learning_rate and max_train_steps
    if(skip_warmup):
        #because we are simply continuing pretraining after clearing ram (i.e. 2nd pass on post_train_data_list)
        config['num_warmup_steps']=0
    else:
        config['num_warmup_steps']=int(np.max([int(max_train_steps/2),1]))

    print("num_warmup_steps:", config['num_warmup_steps'])
    config['max_train_steps'] = max_train_steps*3
    print("max_train_steps:", config['max_train_steps'])
    config['save_checkpoint_steps']=int(np.max([1, max_train_steps//10]))
    print("save_checkpoint_steps",config['save_checkpoint_steps'])
    print("full_epoch_steps:", config['max_train_steps'])

    #reload args
    #args = Namespace(**config)
    #set_seed(args.seed)

    check_patience_after_steps = int(np.max([config['max_train_steps'] // 3,1]))

    # Prepare the optimizer and learning rate scheduler
    optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=config['num_warmup_steps'],
                             num_training_steps=config['max_train_steps'],)

    train_dataset = create_dataset(tokenizer, unique_elements(train_data_list_to_use), chars_per_token)
    train_dataloader = create_dataloader(train_dataset)

    eval_dataset = create_dataset(tokenizer, unique_elements(eval_data_list_to_use), chars_per_token)
    eval_dataloader = create_dataloader(eval_dataset)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

    break_training = False

    progress_bar1 = tqdm(total=check_patience_after_steps, desc="Patience Progress", ncols=100)
    progress_bar2 = tqdm(total=config['max_train_steps'], desc="Total Progress", ncols=100)

    model.train()
    while break_training == False:
        for step, batch in enumerate(train_dataloader, start=1):
            loss = model(batch, labels=batch).loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                progress_bar1.update(1)
                progress_bar2.update(1)

                if completed_steps % config['save_checkpoint_steps'] == 0:
                    eval_loss, perplexity = evaluate(model, eval_dataloader, accelerator)

                    print("\tstep:", step, args.gradient_accumulation_steps, completed_steps, "\tloss:",loss, "\teval loss:", eval_loss, "\tperplexity:", perplexity)

                    #if cmd_args.enable_logging:
                        #log_data = [cmd_args.learning_rate, config['num_warmup_steps'], cmd_args.gradient_accumulation_steps, cmd_args.weight_decay, cmd_args.max_norm, perplexity, eval_loss]
                        #log_to_csv(log_file, log_data)

                    if perplexity < best_perplexity:
                        print("perplexity improved", step, completed_steps, perplexity)
                        best_perplexity = perplexity
                        best_model_state = accelerator.unwrap_model(model).state_dict()
                        consecutive_failures = 0
                    else:
                        print("perplexity failed to improve: ", step, completed_steps, perplexity, consecutive_failures)
                        if completed_steps >= check_patience_after_steps:
                            consecutive_failures += 1

                    if max_completed_steps == None:
                        if consecutive_failures >= 3:
                            break_training = True
                            print("Early stopping: perplexity no longer improving, loading best model.")
                        if stored_models:  # Check if stored_models is not empty
                            best_model_state = min(stored_models, key=lambda x: x[0])[1]
                            model.load_state_dict(best_model_state)

                        if (completed_steps >= check_patience_after_steps):
                            if (perplexity <= desired_perplexity):
                                print("Early stopping: perplexity below threshold.", perplexity)
                                break_training = True
                    elif completed_steps > (max_completed_steps*config['max_train_steps']):
                        break_training = True

                    if completed_steps > config['max_train_steps']:
                        break_training = True
                        break
                    if os.path.exists('stop_training.txt'):
                        print("Stop file detected. Exiting and loading best model.")
                        model.load_state_dict(best_model_state)
                        os.remove('stop_training.txt')  # Remove stop file after exiting the loop
                        break_training = True
                if break_training:
                    print("steps:", step, "completed steps:", completed_steps, "max_train_steps:", config['max_train_steps'])
                    break

                model.train()

    progress_bar1.close()
    progress_bar2.close()

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(path)

    return completed_steps/config['max_train_steps']

def clear_model(accelerator, model, path):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(path)
    del(accelerator)
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return GPTNeoForCausalLM.from_pretrained(path)#.to(device)
