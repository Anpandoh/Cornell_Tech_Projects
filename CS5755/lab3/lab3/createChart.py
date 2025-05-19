import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# List of CSV files to process

def createChart(operationFiles):
    # Initialize a dictionary to store the sum of values and count of occurrences for each area in each file
    operationFiles.reverse()
    operationData = {csv_file: defaultdict(lambda: {'sum': 0, 'count': 0}) for csv_file in operationFiles}


    # Read and process each CSV file
    for csv_file in operationFiles:
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the first line (header)
            for row in reader:
                if row and not row[0].startswith('#') and not row[0].startswith('Area') and not row[0].startswith('MUX'):
                    area = row[0]
                    value = float(row[1])
                    operationData[csv_file][area]['sum'] += value
                    operationData[csv_file][area]['count'] += 1




    # Calculate the average values for each area in each file
    averageoperation = {csv_file[:-4]: {area: values['sum'] / values['count'] for area, values in file_data.items()} for csv_file, file_data in operationData.items()}

    # Extract CSV file names and areas
    operationFiles = list(averageoperation.keys())

    # Desired order of areas
    desired_order = ['Frontend_Bound', 'Bad_Speculation', 'Backend_Bound', 'Retiring']

    # Prepare data for plotting
    data = []
    for file in operationFiles:
        file_data = averageoperation[file]
        data.append([file_data.get(area, 0) for area in desired_order])

    # Convert data to numpy array for easier manipulation
    data = np.array(data)

    # Create a horizontal segmented bar chart
    fig, ax = plt.subplots()

    # Define the bar positions
    bar_positions = np.arange(len(operationFiles))

    # Plot each segment of the bar
    bottom = np.zeros(len(operationFiles))
    for i, area in enumerate(desired_order):
        ax.barh(bar_positions, data[:, i], left=bottom, label=area)
        bottom += data[:, i]

    # Add labels and title
    ax.set_xlabel('Pipeline Bottleneck Breakdown %')
    ax.set_ylabel('Operation')
    ax.set_title('Top-down Analysis')
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(operationFiles)
    ax.legend(title='Areas')

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig('chart.png')
    plt.show()

operationFiles = ['matmul.csv', 'matmul_blocking.csv','matmul_CSR_01.csv', 'matmul_CSR_10.csv', 'matmul_CSR_50.csv', 'matmul_CSR_75.csv', 'matmul_CSR_100.csv']
createChart(operationFiles)
