#!/bin/bash

# Array of possible sizes


command=(lab1)
# Run the command 10 times for each size
for i in {1..10}; do
    # Construct the output file name based on the size
    output_file="test_relu.csv"
    
    # Create a temporary file for the current iteration
    temp_file=$(mktemp)
    
    # Run the command with the selected size and output to the temporary file
    python3 /classes/ece5755/pmu-tools/toplev.py --core S0-C0 -l1 -v --no-desc --force-cpu spr ./lab1 -x, -o $temp_file
    
    # Append a spacer and the contents of the temporary file to the main output file
    echo -e "\n# Iteration $i\n" >> $output_file
    cat $temp_file >> $output_file
    
    # Remove the temporary file
    rm $temp_file
done
