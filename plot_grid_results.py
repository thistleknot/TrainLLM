import csv
import matplotlib.pyplot as plt

def read_hyperparameter_combinations(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip the header row
        data = [row for row in reader]
    return data

def plot_csv_data(log_file, hyperparameter_combinations):
    # Read the CSV data
    with open(log_file, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip the header row
        data = [row for row in reader]

    # Extract the completed_steps, perplexity, and eval_loss values
    completed_steps = [int(row[2]) for row in data]
    perplexity = [float(row[5]) for row in data]
    eval_loss = [float(row[6]) for row in data]

    # Find the log file index from the log_file name
    log_file_index = int(log_file.split("_")[-1].split(".")[0])

    # Get the hyperparameter combination for this log file
    hyperparams = hyperparameter_combinations[log_file_index]

    # Create a line plot
    plt.plot(completed_steps, perplexity, label="Perplexity")
    plt.plot(completed_steps, eval_loss, label="Eval Loss")

    # Set axis labels and title
    plt.xlabel("Completed Steps")
    plt.ylabel("Value")
    title = f"Perplexity and Eval Loss for Different Checkpoints (Log File {log_file_index})"
    title += f"\nLR: {hyperparams[1]}, Warmup: {hyperparams[2]}, GradAccum: {hyperparams[3]}, WeightDecay: {hyperparams[4]}, MaxNorm: {hyperparams[5]}"
    plt.title(title)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

hyperparameter_combinations = read_hyperparameter_combinations("hyperparameter_combinations.csv")
plot_csv_data("perplexity_logs_0.csv", hyperparameter_combinations)
