#include <assert.h>
#include <stdio.h>
#include <stdlib.h>







void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return i + 1;
}

void quickSort(int *arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int *mysort(int *input, int n) {
    quickSort(input, 0, n-1);

    return input;
}


// verify() checks that the array is sorted
// DO NOT call verify() when profiling your code 
int verify(int *input, int n) {
    for (int i = 1; i < n; i++) {
        assert(input[i] >= input[i-1]);
    }
    printf("array is correctly sorted \n");
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("Usage: ./mysort <input_size> <input_file>\n");
        return 1;
    }
    int n = atoi(argv[1]);
    int *input = (int *) malloc(sizeof(int)*n);
    int *output; // you will need to allocate memory for output if not sorting in place
    FILE *fp = fopen(argv[2], "r");
    
    for (int i = 0; i < n; i++) {
        fscanf(fp, "%d,", &input[i] );
    }

    output = mysort(input, n);

    // verify(output, n); // keep this line commented out when profiling

    free(input);
    return 0;
}