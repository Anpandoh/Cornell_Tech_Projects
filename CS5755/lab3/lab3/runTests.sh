# Run the command 5 times
total_time=0

for i in {1..5}; do
    # Construct the output file name
    output_file="matmul_CSR_100.csv"

    # Create a temporary file for the current iteration
    temp_file=$(mktemp)
    echo "iteration $i"
    # python3 /classes/ece5755/pmu-tools/toplev.py --core S0-C0 -l1 -v --no-desc --force-cpu spr ./lab1 -x, -o $temp_file

    # Run the command with the selected size and capture only the elapsed time
    raw_time=$(/usr/bin/time -f "%e" sh -c "python3 /classes/ece5755/pmu-tools/toplev.py --core S0-C0 -l1 -v --no-desc --force-cpu spr ./proflab3 -x, -o $temp_file" 2>&1 | awk '/^[0-9]/ {print $1}')
    
    # Ensure we have a valid numeric elapsed time
    if [[ $raw_time =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        # Add the elapsed time to the total time
        total_time=$(echo "$total_time + $raw_time" | bc)
    else
        echo "Warning: Invalid elapsed time captured: $raw_time"
    fi
    
    # Append a spacer and the contents of the temporary file to the main output file
    echo -e "\n# Iteration $i\n" >> $output_file
    cat $temp_file >> $output_file
    
    # Remove the temporary file
    rm $temp_file
done

# Calculate the average elapsed time
average_time=$(echo "scale=2; $total_time / 5" | bc)
echo "Average time: $average_time seconds"