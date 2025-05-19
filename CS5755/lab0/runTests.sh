#!/bin/bash

# Array of possible sizes
sizes=(100 1000 5000)
commands=(bubble mysort)

# Run the command 10 times for each size
for command in "${commands[@]}"; do
    for size in "${sizes[@]}"; do
        for i in {1..10}; do
            # Construct the output file name based on the size
            output_file="${command}${size}.csv"
            
            # Create a temporary file for the current iteration
            temp_file=$(mktemp)
            
            # Run the command with the selected size and output to the temporary file
            python3 /classes/ece5755/pmu-tools/toplev.py --core S0-C0 -l1 -v --no-desc --force-cpu spr ./$command $size input_$size -x, -o $temp_file
            
            # Append a spacer and the contents of the temporary file to the main output file
            echo -e "\n# Iteration $i\n" >> $output_file
            cat $temp_file >> $output_file
            
            # Remove the temporary file
            rm $temp_file
        done
    done
done
