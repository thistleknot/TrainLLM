# gridsearch.py
from common_imports import *
from functions import save_hyperparameter_combinations

sample = os.environ.get("sample")
sample_size = os.environ.get("sample_size")

# Define the hyperparameter values you want to try
learning_rates = [3e-5, 2e-5, 1e-5, 5e-6]
#num_warmup_steps_options = ["sample_size", "sample_size/2"]
gradient_accumulation_steps = [8, 16]
weight_decays = [0.1, 0.25]
max_norms = [0.25, 0.5, 1.0]

hyperparameter_combinations = list(itertools.product(learning_rates, num_warmup_steps_options, gradient_accumulation_steps, weight_decays, max_norms))

# Iterate over all the combinations and call pretrain_neo.py with the corresponding hyperparameters
save_hyperparameter_combinations(hyperparameter_combinations, "hyperparameter_combinations.csv")

for index, combination in enumerate(hyperparameter_combinations):
    learning_rate, num_warmup_steps, gradient_accumulation_step, weight_decay, max_norm = combination

    # Update num_warmup_steps based on its value
    if num_warmup_steps == "sample_size":
        num_warmup_steps = sample_size  # Replace sample_size with the actual value
    elif num_warmup_steps == "sample_size/2":
        num_warmup_steps = sample_size // 2  # Replace sample_size with the actual value

    # Call pretrain_neo.py with the current hyperparameter combination and log file index
    subprocess.run(["python", "pretrain_neo.py",
                    "--learning_rate", str(learning_rate),
                    #"--num_warmup_steps", str(num_warmup_steps),
                    "--gradient_accumulation_steps", str(gradient_accumulation_step),
                    "--weight_decay", str(weight_decay),
                    "--max_norm", str(max_norm),
                    "--log_file_index", str(index),
                    "--enable_logging"])  # Add this line])
