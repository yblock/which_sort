import timeit
import random
from typing import List,Dict, Any, Callable, Tuple


"""
- Quick Sort - 
This quick_sort() function is a recursive implementation of the quicksort algorithm that sorts an input array in
ascending order. The function first checks if the length of the array is 1 or less, in which case it returns the
array. If the array has more than one element, the function selects a pivot point as the middle element and creates
three subarrays - one for elements less than the pivot, one for elements equal to the pivot, and one for elements
greater than the pivot. The function then recursively calls itself on the left and right subarrays, and concatenates
the results with the middle subarray to produce the final sorted array. 
"""
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

""" 
- Merge Sort -
This merge sort algorithm takes an array as input, and recursively divides the array into halves
until the base case is reached where the length of the array is 1 or less. It then uses the ms_merge() helper function
to merge the halves back together in sorted order. The ms_merge() function compares the left and right lists and
appends the smallest value to the result list until all values have been appended. The final sorted result is then
returned. 
"""
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    return ms_merge(merge_sort(left), merge_sort(right))
# Merge function for merge_sort. Compares left and right list
def ms_merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result

"""
- Timsort -
This is already Python's built in sort() method, but I wanted to try recreating it
Timsort splits the list into small runs, sorts them using insertion sort, and merges them together
"""
def timsort(arr: List, run: int = 32) -> List:
    
    def insertion_sort(arr: List, left: int, right: int) -> None:
    # Sorts the elements in the range [left, right] of the input list arr using the insertion sort algorithm.
    # The function works by iterating over each element in the range and inserting it into its correct position
    # relative to the other elements in the range. The algorithm maintains a sorted subarray to the left of
    # the current element, and shifts each element in this subarray to the right until the correct position for
    # the current element is found. The function modifies the input list arr in place and returns None.
        for i in range(left + 1, right + 1):
            key = arr[i]
            j = i - 1
            while j >= left and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def merge(arr: List, l: int, m: int, r: int) -> None:
        len1, len2 = m - l + 1, r - m
        left, right = arr[l:l + len1], arr[m + 1:m + 1 + len2]
        i = j = 0
        k = l

        while i < len1 and j < len2:
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len1 and k < len(arr):
            arr[k] = left[i]
            k += 1
            i += 1

        while j < len2 and k < len(arr):
            arr[k] = right[j]
            k += 1
            j += 1

    n = len(arr)
    for i in range(0, n, run):
        insertion_sort(arr, i, min(i + run - 1, n - 1))

    size = run
    while size < n:
        for left in range(0, n, 2 * size):
            mid = left + size - 1
            right = min(left + 2 * size - 1, n - 1)
            merge(arr, left, mid, right)
        size *= 2

    return arr


# Analyzes list characteristics and returns a dictionary of size, number of unique_values, the max_value, and the min_value
def analyze_dataset(data: List) -> Dict[str, Any]:
    characteristics = {
        'size': len(data),
        'unique_values': len(set(data)),
        'max_value': max(data),
        'min_value': min(data),
    }

    return characteristics

# Benchmarks sorting algorithms and stores results in a list of tuples containing algorithm name and sorting time
def benchmark_sorting_algorithms(data: List, algorithms: List[Callable]) -> List[Tuple[str, float]]:
    results = []
    print("Benchmarking...")
    for algorithm in algorithms:
        name = algorithm.__name__
        wrapper = lambda: algorithm(list(data))
        time = timeit.timeit(wrapper, number=1)
        results.append((name, time))
    return results

# This function selects the best sorting algorithm from available algorithms based on the characteristics analyze_dataset().
def best_sorting_algorithm(data: List) -> Callable:
    characteristics = analyze_dataset(data)
    candidates = [quick_sort, merge_sort, timsort]
    benchmark_results = benchmark_sorting_algorithms(data, candidates)

    # Prioritize Tim Sort if there are many duplicate values
    if characteristics['unique_values'] / characteristics['size'] < 0.5:
        return timsort

    best_algorithm = min(benchmark_results, key=lambda x: x[1])[0]
    print(best_algorithm, " selected.")

    # Map the algorithm names to their functions
    algorithms = {
        'quick_sort': quick_sort,
        'merge_sort': merge_sort,
        'tim_sort': timsort
    }

    return algorithms[best_algorithm]


def main():
    data_count = random.randint(1, 1000000)
    # when randints are restricted below 1M, timsort is almost always the fastest option. Above that, quicksort takes the cake!
    data = [random.randint(0, 1000000) for _ in range(data_count)]
    sorting_algorithm = best_sorting_algorithm(data)
    sorted_data = sorting_algorithm(data)
    
    print("Sorted data:", sorted_data)
    print("Best sorting algorithm:", sorting_algorithm.__name__)
    print("Number of data points: ", len(data))

if __name__ == "__main__":
    main()