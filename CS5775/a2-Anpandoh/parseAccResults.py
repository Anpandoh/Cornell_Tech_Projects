import re
import matplotlib.pyplot as plt
import numpy as np

Accuracy = """
Val Acc Epoch 1 = 83.57%, Train Acc Epoch 1 = 64.48%, Train loss = 1.06

Valid epoch: 1	Accuracy: 1100/1333 (82.52%)

Val Acc Epoch 2 = 84.77%, Train Acc Epoch 2 = 79.32%, Train loss = 0.562

Valid epoch: 2	Accuracy: 1136/1333 (85.22%)

Val Acc Epoch 3 = 85.97%, Train Acc Epoch 3 = 82.19%, Train loss = 0.489

Valid epoch: 3	Accuracy: 1156/1333 (86.72%)

Val Acc Epoch 4 = 87.62%, Train Acc Epoch 4 = 82.45%, Train loss = 0.47

Valid epoch: 4	Accuracy: 1164/1333 (87.32%)

Val Acc Epoch 5 = 88.07%, Train Acc Epoch 5 = 83.7%, Train loss = 0.446

Valid epoch: 5	Accuracy: 1171/1333 (87.85%)

Val Acc Epoch 6 = 88.22%, Train Acc Epoch 6 = 83.9%, Train loss = 0.439

Valid epoch: 6	Accuracy: 1174/1333 (88.07%)

Val Acc Epoch 7 = 88.67%, Train Acc Epoch 7 = 84.08%, Train loss = 0.427

Valid epoch: 7	Accuracy: 1194/1333 (89.57%)

Val Acc Epoch 8 = 89.05%, Train Acc Epoch 8 = 85.35%, Train loss = 0.405

Valid epoch: 8	Accuracy: 1184/1333 (88.82%)

Val Acc Epoch 9 = 88.97%, Train Acc Epoch 9 = 85.51%, Train loss = 0.401

Valid epoch: 9	Accuracy: 1196/1333 (89.72%)

Val Acc Epoch 10 = 88.67%, Train Acc Epoch 10 = 84.84%, Train loss = 0.401

Valid epoch: 10	Accuracy: 1193/1333 (89.50%)

Val Acc Epoch 11 = 88.22%, Train Acc Epoch 11 = 84.98%, Train loss = 0.41

Valid epoch: 11	Accuracy: 1177/1333 (88.30%)

Val Acc Epoch 12 = 89.2%, Train Acc Epoch 12 = 85.46%, Train loss = 0.397

Valid epoch: 12	Accuracy: 1190/1333 (89.27%)

Val Acc Epoch 13 = 88.15%, Train Acc Epoch 13 = 85.58%, Train loss = 0.392

Valid epoch: 13	Accuracy: 1183/1333 (88.75%)

Val Acc Epoch 14 = 89.95%, Train Acc Epoch 14 = 85.72%, Train loss = 0.392

Valid epoch: 14	Accuracy: 1171/1333 (87.85%)

Val Acc Epoch 15 = 89.2%, Train Acc Epoch 15 = 86.22%, Train loss = 0.386

Valid epoch: 15	Accuracy: 1196/1333 (89.72%)

Val Acc Epoch 16 = 89.35%, Train Acc Epoch 16 = 85.94%, Train loss = 0.378

Valid epoch: 16	Accuracy: 1192/1333 (89.42%)

Val Acc Epoch 17 = 88.75%, Train Acc Epoch 17 = 86.11%, Train loss = 0.385

Valid epoch: 17	Accuracy: 1186/1333 (88.97%)

Val Acc Epoch 18 = 90.1%, Train Acc Epoch 18 = 86.27%, Train loss = 0.374

Valid epoch: 18	Accuracy: 1203/1333 (90.25%)

Val Acc Epoch 19 = 89.72%, Train Acc Epoch 19 = 86.2%, Train loss = 0.368

Valid epoch: 19	Accuracy: 1216/1333 (91.22%)

Val Acc Epoch 20 = 90.62%, Train Acc Epoch 20 = 86.42%, Train loss = 0.37

Valid epoch: 20	Accuracy: 1208/1333 (90.62%)

Val Acc Epoch 21 = 90.1%, Train Acc Epoch 21 = 86.32%, Train loss = 0.37

Valid epoch: 21	Accuracy: 1191/1333 (89.35%)

Val Acc Epoch 22 = 91.45%, Train Acc Epoch 22 = 86.94%, Train loss = 0.36

Valid epoch: 22	Accuracy: 1207/1333 (90.55%)

Val Acc Epoch 23 = 89.42%, Train Acc Epoch 23 = 86.38%, Train loss = 0.375

Valid epoch: 23	Accuracy: 1204/1333 (90.32%)

Val Acc Epoch 24 = 89.27%, Train Acc Epoch 24 = 86.09%, Train loss = 0.373

Valid epoch: 24	Accuracy: 1190/1333 (89.27%)

Val Acc Epoch 25 = 90.7%, Train Acc Epoch 25 = 86.44%, Train loss = 0.371

Valid epoch: 25	Accuracy: 1201/1333 (90.10%)

Val Acc Epoch 26 = 90.85%, Train Acc Epoch 26 = 86.24%, Train loss = 0.368

Valid epoch: 26	Accuracy: 1209/1333 (90.70%)

Val Acc Epoch 27 = 90.17%, Train Acc Epoch 27 = 87.18%, Train loss = 0.352

Valid epoch: 27	Accuracy: 1198/1333 (89.87%)

Val Acc Epoch 28 = 90.02%, Train Acc Epoch 28 = 87.16%, Train loss = 0.354

Valid epoch: 28	Accuracy: 1201/1333 (90.10%)

Val Acc Epoch 29 = 91.07%, Train Acc Epoch 29 = 87.25%, Train loss = 0.352

Valid epoch: 29	Accuracy: 1204/1333 (90.32%)

Val Acc Epoch 30 = 90.17%, Train Acc Epoch 30 = 86.88%, Train loss = 0.358

Valid epoch: 30	Accuracy: 1210/1333 (90.77%)

Val Acc Epoch 31 = 90.55%, Train Acc Epoch 31 = 87.08%, Train loss = 0.35

Valid epoch: 31	Accuracy: 1210/1333 (90.77%)

Val Acc Epoch 32 = 91.3%, Train Acc Epoch 32 = 86.7%, Train loss = 0.358

Valid epoch: 32	Accuracy: 1224/1333 (91.82%)

Val Acc Epoch 33 = 91.45%, Train Acc Epoch 33 = 86.84%, Train loss = 0.348

Valid epoch: 33	Accuracy: 1219/1333 (91.45%)

Val Acc Epoch 34 = 91.52%, Train Acc Epoch 34 = 87.14%, Train loss = 0.349

Valid epoch: 34	Accuracy: 1202/1333 (90.17%)

Val Acc Epoch 35 = 90.47%, Train Acc Epoch 35 = 87.5%, Train loss = 0.34

Valid epoch: 35	Accuracy: 1220/1333 (91.52%)

Val Acc Epoch 36 = 90.77%, Train Acc Epoch 36 = 86.98%, Train loss = 0.361

Valid epoch: 36	Accuracy: 1207/1333 (90.55%)

Val Acc Epoch 37 = 90.1%, Train Acc Epoch 37 = 87.72%, Train loss = 0.343

Valid epoch: 37	Accuracy: 1221/1333 (91.60%)

Val Acc Epoch 38 = 91.07%, Train Acc Epoch 38 = 87.7%, Train loss = 0.341

Valid epoch: 38	Accuracy: 1211/1333 (90.85%)

Val Acc Epoch 39 = 91.6%, Train Acc Epoch 39 = 87.85%, Train loss = 0.332

Valid epoch: 39	Accuracy: 1218/1333 (91.37%)

Val Acc Epoch 40 = 91.9%, Train Acc Epoch 40 = 87.68%, Train loss = 0.342

Valid epoch: 40	Accuracy: 1232/1333 (92.42%)

Val Acc Epoch 41 = 91.97%, Train Acc Epoch 41 = 87.18%, Train loss = 0.346

Valid epoch: 41	Accuracy: 1222/1333 (91.67%)

Val Acc Epoch 42 = 91.07%, Train Acc Epoch 42 = 87.86%, Train loss = 0.336

Valid epoch: 42	Accuracy: 1215/1333 (91.15%)

Val Acc Epoch 43 = 91.15%, Train Acc Epoch 43 = 87.23%, Train loss = 0.35

Valid epoch: 43	Accuracy: 1218/1333 (91.37%)

Val Acc Epoch 44 = 91.97%, Train Acc Epoch 44 = 87.95%, Train loss = 0.332

Valid epoch: 44	Accuracy: 1226/1333 (91.97%)

Val Acc Epoch 45 = 92.2%, Train Acc Epoch 45 = 87.71%, Train loss = 0.335

Valid epoch: 45	Accuracy: 1218/1333 (91.37%)

Val Acc Epoch 46 = 92.57%, Train Acc Epoch 46 = 87.68%, Train loss = 0.342

Valid epoch: 46	Accuracy: 1234/1333 (92.57%)

Val Acc Epoch 47 = 92.2%, Train Acc Epoch 47 = 87.68%, Train loss = 0.336

Valid epoch: 47	Accuracy: 1233/1333 (92.50%)

Val Acc Epoch 48 = 91.97%, Train Acc Epoch 48 = 88.12%, Train loss = 0.328

Valid epoch: 48	Accuracy: 1212/1333 (90.92%)

Val Acc Epoch 49 = 90.77%, Train Acc Epoch 49 = 87.78%, Train loss = 0.335

Valid epoch: 49	Accuracy: 1208/1333 (90.62%)

Val Acc Epoch 50 = 92.2%, Train Acc Epoch 50 = 87.77%, Train loss = 0.337

Valid epoch: 50	Accuracy: 1220/1333 (91.52%)

Test Acc = 90.57%

"""


def parse_accuracy_data(text):
    # Regular expressions to extract data
    val_acc_pattern = r"Val Acc Epoch (\d+) = ([\d.]+)%"
    train_acc_pattern = r"Train Acc Epoch (\d+) = ([\d.]+)%"
    
    # Extract validation accuracy data
    val_data = re.findall(val_acc_pattern, text)
    val_epochs = [int(epoch) for epoch, _ in val_data]
    val_accuracies = [float(acc) for _, acc in val_data]
    
    # Extract training accuracy data
    train_data = re.findall(train_acc_pattern, text)
    train_epochs = [int(epoch) for epoch, _ in train_data]
    train_accuracies = [float(acc) for _, acc in train_data]
    
    return val_epochs, val_accuracies, train_epochs, train_accuracies

def plot_accuracy_curves(text):
    val_epochs, val_accuracies, train_epochs, train_accuracies = parse_accuracy_data(text)
    
    plt.figure(figsize=(12, 6))
    plt.plot(val_epochs, val_accuracies, 'b-', label='Validation Accuracy')
    plt.plot(train_epochs, train_accuracies, 'r-', label='Training Accuracy')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('FP32 Model Accuracy Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_curves.png')
    plt.show()

def plot_quantization_comparison(post_training_quant, qat_int):
    """
    Plot bar charts comparing post-training quantization and QAT at different bit widths
    """
    # Extract the data
    bit_widths = list(post_training_quant.keys())
    post_training_accs = list(post_training_quant.values())
    qat_accs = [qat_int[bw] for bw in bit_widths]
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Set the width of the bars
    bar_width = 0.35
    
    # Set the positions of the bars on the x-axis
    r1 = np.arange(len(bit_widths))
    r2 = [x + bar_width for x in r1]
    
    # Create the bars
    plt.bar(r1, post_training_accs, width=bar_width, color='skyblue', edgecolor='black', 
           label='Post-Training Quantization')
    plt.bar(r2, qat_accs, width=bar_width, color='orange', edgecolor='black',
           label='Quantization-Aware Training')
    
    # Add labels, title and legend
    plt.xlabel('Bit Width', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Accuracy Comparison: Post-Training Quantization vs QAT')
    plt.xticks([r + bar_width/2 for r in range(len(bit_widths))], bit_widths)
    plt.legend()
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels on top of each bar
    for i, v in enumerate(post_training_accs):
        plt.text(i - 0.05, v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    for i, v in enumerate(qat_accs):
        plt.text(i + bar_width - 0.05, v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig('quantization_comparison.png')
    plt.show()

# Use the Accuracy string from your file
plot_accuracy_curves(Accuracy)

# Plot the quantization comparison
post_training_quant_fp32_int = {2:19.883, 4:87.135, 6: 88.743, 8:88.523}
qat_int = {2: 30.629, 4: 87.573, 6: 89.035, 8: 90.278}
plot_quantization_comparison(post_training_quant_fp32_int, qat_int)


# structuredPruningParams = {0.2: 12490, 0.4: 10409, 0.6: 6247, 0.8: 4166, 0.9: 2085}
structuredPruningAccuracyNonFineTuned = {12490: 56.26, 10409: 61.97, 6247: 54.09, 4166: 42.01, 2085: 32.03}
structuredPruningAccuracyFineTuned = {12490: 79.19, 10409: 78.99, 6247: 64.74, 4166: 60.69, 2085: 40.21}

structuredPruningFlopFineTuned = {507004: 79.19, 422504: 78.99, 253504: 64.74, 169004: 60.69, 84504: 40.21}
structuredPruningCPULatencyFineTuned = {2199.06: 79.19, 2487.05: 78.99, 2269.63: 64.74, 2499.61: 60.69, 2400.02: 40.21}
structuredPruningMCULatencyFineTuned = {80.21: 79.19, 76.2: 78.99, 65.33: 64.74, 53.88: 60.69, 50.44: 40.21}

def plot_pruning_latency_accuracy(fine_tuned):
    # Extract latencies and accuracies from the dictionary
    latencies = list(fine_tuned.keys())
    fine_tuned_accs = list(fine_tuned.values())
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the curve
    plt.plot(fine_tuned_accs, latencies, 'r-o', label='Fine-Tuned Accuracy')
    
    # Add labels, title and legend
    plt.ylabel('CPU Latency (ms)', fontweight='bold')
    plt.xlabel('Accuracy (%)', fontweight='bold')
    plt.title('Structured Pruning Accuracy vs MCU Latency on MCU')
    plt.legend()
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis to start at 0
    # plt.ylim(bottom=0)
    # plt.ylim(top=3000)

    
    # Add value labels on top of each point
    for i, v in enumerate(latencies):
        plt.text(fine_tuned_accs[i], v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig('pruning_latency_accuracy.png')
    plt.show()

# Plot the pruning latency accuracy curve
plot_pruning_latency_accuracy(structuredPruningMCULatencyFineTuned)




def plot_pruning_flop_accuracy(fine_tuned):
    # Convert FLOPs to KFLOPs
    kflops = [flop / 1e3 for flop in fine_tuned.keys()]
    fine_tuned_accs = list(fine_tuned.values())
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the curve
    plt.plot(kflops, fine_tuned_accs, 'r-o', label='Fine-Tuned Accuracy')
    
    # Add labels, title and legend
    plt.xlabel('Number of KFLOPs', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Structured Pruning Accuracy vs KFLOPs')
    plt.legend()
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of each point
    for i, v in enumerate(fine_tuned_accs):
        plt.text(kflops[i], v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig('pruning_flop_accuracy.png')
    plt.show()

# Plot the pruning FLOP accuracy curve
plot_pruning_flop_accuracy(structuredPruningFlopFineTuned)



def plot_pruning_accuracy(non_fine_tuned, fine_tuned):
    # Extract the data
    params = list(non_fine_tuned.keys())
    non_fine_tuned_accs = list(non_fine_tuned.values())
    fine_tuned_accs = list(fine_tuned.values())
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the curves
    plt.plot(params, non_fine_tuned_accs, 'b-o', label='Non-Fine-Tuned Accuracy')
    plt.plot(params, fine_tuned_accs, 'r-o', label='Fine-Tuned Accuracy')
    
    # Add labels, title and legend
    plt.xlabel('Number of Parameters', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Structured Pruning Accuracy vs Parameters')
    plt.legend()
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of each point
    for i, v in enumerate(non_fine_tuned_accs):
        plt.text(params[i], v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    for i, v in enumerate(fine_tuned_accs):
        plt.text(params[i], v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    plt.tight_layout()
    plt.savefig('pruning_accuracy_parameters.png')
    plt.show()

# Plot the pruning accuracy curves
plot_pruning_accuracy(structuredPruningAccuracyNonFineTuned, structuredPruningAccuracyFineTuned)

# unstructuredPruning
def plot_unstructured_pruning_accuracy():
    # Data for unstructured pruning
    params = [14987, 13322, 11656, 9991, 8326, 6661, 4995, 3330, 1665]
    Non_Finetuned_Accuracy = [90.1, 90.55, 90.1, 90.47, 90.4, 89.8, 82.82, 45.39, 32.78]
    Finetuned_Accuracy = [91.15, 90.62, 91.45, 90.62, 90.62, 89.57, 86.42, 74.42, 50.11]
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the curves
    plt.plot(params, Non_Finetuned_Accuracy, 'b-o', label='Non-Finetuned Accuracy')
    plt.plot(params, Finetuned_Accuracy, 'r-o', label='Finetuned Accuracy')
    
    # Add labels, title and legend
    plt.xlabel('Number of Parameters', fontweight='bold')
    plt.ylabel('Accuracy (%)', fontweight='bold')
    plt.title('Unstructured Pruning Accuracy vs Number of Parameters')
    plt.legend()
    
    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of each point
    # for i, v in enumerate(Non_Finetuned_Accuracy):
    #     plt.text(params[i], v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    # for i, v in enumerate(Finetuned_Accuracy):
    #     plt.text(params[i], v + 0.5, f"{v:.2f}", fontsize=9, rotation=0)
    
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig('unstructured_pruning_accuracy.png')
    plt.show()

# Plot the unstructured pruning accuracy curves
plot_unstructured_pruning_accuracy()


#tfline runtime breakdown
def plot_runtime_breakdown():
    # Data for the pie chart
    labels = ['Preprocessing', 'Neural Network', 'Postprocessing']
    times = [22, 88, 0]  # Average times
    
    # Set up the figure
    plt.figure(figsize=(8, 8))
    
    # Plot the pie chart
    plt.pie(times, labels=labels, startangle=140, colors=['skyblue', 'orange', 'lightgreen'])
    
    # Add title
    plt.title('Runtime Breakdown')
    
    # Add a legend with the actual times
    plt.legend([f"{label}: {time} ms" for label, time in zip(labels, times)], loc="best")
    
    plt.tight_layout()
    plt.savefig('runtime_breakdown.png')
    plt.show()

# Plot the runtime breakdown pie chart
plot_runtime_breakdown()
