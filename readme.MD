# Sorting Algorithm Fun

This Python script is an experimentation of 3 popular sorting algorithms (quick sort, merge sort, and timsort) to determine the fastest sorting algorithm for a given dataset of integers.


## Getting Started

To get started with this project, you will need to ensure you have Python installed on your computer. Once Python is installed, you can clone the repository and run the following command to install the required packages:


`pip install -r requirements.txt`

## Usage

To use the script, execute the following command:

`python sorting_algorithm_benchmark.py`

This will generate a random dataset of integers and print the sorted data, the best sorting algorithm, and the number of data points. You can modify the dataset by changing the `data_count` and `data` variable in main().


## What it's doing under the hood

The script will select the best sorting algorithm from the available algorithms based on the characteristics of the dataset. The characteristics analyzed are size, number of unique values, max value, and min value. 


Timsort is prioritized if there are many duplicate values to be sorted. If the unique values are less than half of the size of the dataset, timsort will be selected as the best algorithm beacuse it's really good at efficiently sorting that kind of dataset.


If timsort isn't prioritized, the script will benchmark the performance of the available sorting algorithms and store the results in a list of tuples containing the algorithm name and sorting time.


The results of the sorting algorithm with the fastest sort time will be executed and displayed at the end of the script execution along with how many values were generated for that run.