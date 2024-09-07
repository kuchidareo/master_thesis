import json
import argparse
from collections import defaultdict

loss_diff_threshold = 0.01
accuracy_threshold = 0.96

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process JSON data for convergence detection.')
parser.add_argument('--json_file', type=str, help='Path to the JSON file')
args = parser.parse_args()
print(args)
# Load the JSON data
with open(args.json_file) as f:
    data = json.load(f)

# Group data by fraction_rate
grouped_data = defaultdict(list)
for item in data:
    # grouped_data[item['fraction']].append(item)
    grouped_data[item["num_available"]].append(item)

grouped_data = dict(sorted(grouped_data.items(), reverse=True))
# Define a function to detect convergence
def detect_loss_convergence(data, loss_diff_threshold):
    for i in range(1, len(data)):
        loss_diff = data[i-1]['loss'] - data[i]['loss']
        if loss_diff < loss_diff_threshold:
            return data[i]['round'], data[i]['elapsed_minitues']
    return None, None

# Define a function to detect convergence
def detect_threshold_accuracy(data, accuracy_threshold):
    for item in data:
        accuracy = item['accuracy']
        if accuracy > accuracy_threshold:
            return item['round'], item['elapsed_minitues']
    return None, None

# # Detect the convergence round for each fraction_rate
# for fraction, fraction_data in grouped_data.items():
#     convergence_round, convergence_time = detect_loss_convergence(fraction_data, loss_diff_threshold)
#     print(f"Fraction {fraction} - Loss Convergence round: {convergence_round}")
#     print(f"Fraction {fraction} - Loss Convergence time: {convergence_time}")
    
#     # threshold_accuracy_round, threshold_accuracy_time = detect_threshold_accuracy(fraction_data, accuracy_threshold)
#     # print(f"Fraction {fraction} - Accuracy {accuracy_threshold}% at round: {threshold_accuracy_round}")
#     # print(f"Fraction {fraction} - Accuracy {accuracy_threshold}% at time: {threshold_accuracy_time}")

#     print('------------------------------------------')

# Detect the convergence round for each fraction_rate
for numdevice, fraction_data in grouped_data.items():
    convergence_round, convergence_time = detect_loss_convergence(fraction_data, loss_diff_threshold)
    # print(f"Number of Device {numdevice} - Loss Convergence round: {convergence_round}")
    # print(f"Number of Device {numdevice} - Loss Convergence time: {convergence_time}")
    
    threshold_accuracy_round, threshold_accuracy_time = detect_threshold_accuracy(fraction_data, accuracy_threshold)
    print(f"Number of Device {numdevice} - Accuracy {accuracy_threshold}% at round: {threshold_accuracy_round}")
    print(f"Number of Device {numdevice} - Accuracy {accuracy_threshold}% at time: {threshold_accuracy_time}")

    print('------------------------------------------')