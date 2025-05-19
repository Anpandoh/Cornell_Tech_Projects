import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# List of CSV files to process
bubbleFiles = ['bubble100.csv', 'bubble1000.csv', 'bubble5000.csv']
mysortFiles = ['mysort100.csv', 'mysort1000.csv', 'mysort5000.csv']


# Initialize a dictionary to store the sum of values and count of occurrences for each area in each file
bubbleData = {csv_file: defaultdict(lambda: {'sum': 0, 'count': 0}) for csv_file in bubbleFiles}
mysortData = {csv_file: defaultdict(lambda: {'sum': 0, 'count': 0}) for csv_file in mysortFiles}


# Read and process each CSV file
for csv_file in bubbleFiles:
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line (header)
        for row in reader:
            if row and not row[0].startswith('#') and not row[0].startswith('Area') and not row[0].startswith('MUX'):
                area = row[0]
                value = float(row[1])
                bubbleData[csv_file][area]['sum'] += value
                bubbleData[csv_file][area]['count'] += 1

for csv_file in mysortFiles:
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the first line (header)
        
        for row in reader:
            if row and not row[0].startswith('#') and not row[0].startswith('Area') and not row[0].startswith('MUX'):
                area = row[0]
                value = float(row[1])
                mysortData[csv_file][area]['sum'] += value
                mysortData[csv_file][area]['count'] += 1




# Calculate the average values for each area in each file
averageBubble = {csv_file[:-4]: {area: values['sum'] / values['count'] for area, values in file_data.items()} for csv_file, file_data in bubbleData.items()}
averageMysort = {csv_file[:-4]: {area: values['sum'] / values['count'] for area, values in file_data.items()} for csv_file, file_data in mysortData.items()}

# Extract CSV file names and areas
bubbleFiles = list(averageBubble.keys())
mysortFiles = list(averageMysort.keys())
bubbleAreas = list(next(iter(averageBubble.values())).keys())
mysortAreas = list(next(iter(averageMysort.values())).keys())


combinedAreas = list(set(bubbleAreas + mysortAreas))

# Prepare data for plotting
bubbleValues = np.array([[averageBubble[csv].get(area, 0) for area in combinedAreas] for csv in bubbleFiles])
mysortValues = np.array([[averageMysort[csv].get(area, 0) for area in combinedAreas] for csv in mysortFiles])

# Combine data with a space
combinedFiles = bubbleFiles + [''] + mysortFiles
combinedValues = np.vstack((bubbleValues, np.zeros((1, len(combinedAreas))), mysortValues))  # Add a row of zeros for the space

# Desired order of areas
desired_order = ['Frontend_Bound', 'Bad_Speculation', 'Backend_Bound', 'Retiring']

# Create an index mapping from current order to desired order
index_mapping = [combinedAreas.index(area) for area in desired_order]

# Reorder combinedAreas
combinedAreas = [combinedAreas[i] for i in index_mapping]

# Reorder combinedValues columns
combinedValues = combinedValues[:, index_mapping]


fig, ax = plt.subplots(figsize=(14, 7))

# Plotting combined data
left = np.zeros(len(combinedFiles))
for i, area in enumerate(combinedAreas):
    ax.barh(combinedFiles, combinedValues[:, i], left=left, label=area)
    left += combinedValues[:, i]

ax.set_xlabel('Pipleline Bottleneck Breakdown %')
ax.set_ylabel('Programs')
ax.set_title('Top-down Analysis')
ax.legend(title='Areas')

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig('chart.png')