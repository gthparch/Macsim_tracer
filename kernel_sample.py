import csv
import statistics
from sklearn.cluster import KMeans
import math
import os
import argparse
import math
import random

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        line = next(file)
        while ',' not in line:
            line = next(file)

        column_names = line.strip().split(',')

        # Read the rest of the file as the 2D table
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not ''.join(row).strip():  # Check if the row, when joined and stripped, is empty
                continue  # Skip appending empty rows
            data.append(row)

    return column_names, data

def dur_map(column_names, data):
    ret_map = {}
    for kernel_id, row in enumerate(data):
        # key = "x".join(row[-3].split()) + "/" + "x".join(row[-2].split()) + "/" + row[-1].split("<")[0].strip()
        key = row[-1].split("(")[0].strip()
        if key not in ret_map:
            ret_map[key] = []
        ret_map[key].append((kernel_id, row[column_names.index("Kernel Dur (ns)")]))

    return ret_map

def cluster_1dlist(l_list):
    # Reshape the list to make it suitable for clustering
    data = [[int(x[1])] for x in l_list]

    # Initialize the KMeans model with k=2
    kmeans = KMeans(n_clusters=2)

    # Fit the model to the data
    kmeans.fit(data)

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Group data points based on their cluster labels
    clustered_data = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_data:
            clustered_data[label] = []
        clustered_data[label].append(l_list[i])

    return clustered_data

def cluster_recursive(key, values, threshold, c, kernel_info):
    cluster = cluster_1dlist(values)
    for kkey, value in cluster.items():
        durations = list(map(float, [v[1] for v in value]))
        avg = statistics.mean(durations)
        stdev = statistics.stdev(durations) if len(durations) > 1 else 0
        n_samples = (c * stdev / avg) ** 2 if stdev != 0 else 1
        new_key = key + str(kkey)

        if n_samples < threshold:
            kernel_info[new_key] = n_samples, [v[0] for v in value]
            continue

        cluster_recursive(new_key, value, threshold, c, kernel_info)

def parse(name, threshold, error, min_n):
    if error == 1:
        c = 196
    elif error == 5:
        c = 39.2

    # Example usage:
    nsys = f"{name}.csv" 
    column_names, data = read_csv(nsys)

    nsys_summary = f"{name}_summary.csv"
    column_names_summary, data_summary = read_csv(nsys_summary)

    d_map = dur_map(column_names, data)

    # kernel_info[gridsize + blocksize + kernel name + cluster_ids] = n_samples, [kernel ids]
    kernel_info = {}

    for key, value in d_map.items():
        durations = list(map(float, [v[1] for v in value]))
        avg = statistics.mean(durations)
        stdev = statistics.stdev(durations) if len(durations) > 1 else 0
        n_samples = (c * stdev / avg) ** 2 if stdev != 0 else 1
        
        if n_samples < threshold:
            kernel_info[key] = n_samples, [v[0] for v in value]
            continue
        cluster_recursive(key, value, threshold, c, kernel_info)

    tot_kernel = 0
    for key, value in kernel_info.items():
        tot_kernel += max(math.ceil(value[0]), min_n)
        # print(f"key: {key[:60]}, n_samples: {value[0]}, kernel_ids: {value[1][:math.ceil(value[0])]}")

    ### Kernels to run
    # for key, value in kernel_info.items():
    #     print(f"kernel_ids: {value[1][:math.ceil(value[0])]}")

    duration_sum = sum([int(row[column_names.index("Kernel Dur (ns)")]) for row in data])
    sampled_duration_sum = 0

    sampled_kernels = []

    with open(f"{name}_sampled_kernels_info.txt", "w") as file:
        file.write(f"Kernel sampling information from {name}.nsys-rep.\n")
        file.write(f"Maximum cluster size: {threshold}, Error bound: {error}% with 95% certainty.\n")
        for key, value in kernel_info.items():
            # Cluster_size, Sample_size, Sample_ids

            # from a list value[1], randomly pick math.ceil(value[0]) elements (with duplicates)
            m_min = max(math.ceil(value[0]), min_n)
            sample_ids = random.choices(value[1], k=m_min)
            sample_ids = list(map(int, sample_ids))
            for sample_id in sample_ids:
                sampled_duration_sum += int(data[sample_id][column_names.index("Kernel Dur (ns)")])
                sampled_kernels.append(sample_id)
            sample_ids = " ".join(map(str, sample_ids))
            file.write(f"{len(value[1])} {m_min} {sample_ids}\n")
            
        print(f"Outputs are printed to {name}_sampled_kernels_info.txt") 

    print(f"Original number of kernels: {len(data)}")
    print(f"Total number of kernels to run: {tot_kernel}")
    print(f"Speedup: {duration_sum / sampled_duration_sum}") 
    if args.print_samples:
        print(f"Sampled kernels: {sampled_kernels}")

    return f"{name}_sampled_kernels_info.txt"

def trace_generate(cmd, trace_path, sample_info_file, device_id):
    print(f"Generating trace file...")
    print(f"cmd: {cmd}")
    print(f"trace_path: {trace_path}")
    os.system(f"CUDA_INJECTION64_PATH=/fast_data/echung67/nvbit_release/tools/main/main.so \
        CUDA_VISIBLE_DEVICES={device_id} \
        TRACE_PATH={trace_path} \
        DEBUG_TRACE=0 \
        OVERWRITE=1 \
        SAMPLED_KERNEL_INFO={sample_info_file} \
        {cmd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="Command to profile - Python or CUDA binary.")
    parser.add_argument("--name", default="nsys", help="Name of the nsys output file. ex: --cmd=\"python3 bert_medium.py\"")
    parser.add_argument("--threshold", type=int, default=50, help="Maximum number of kernel samples.")
    parser.add_argument("--error", type=int, default=5, choices=[1, 5], help="Error (%) bound with 95% confidence.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing nsys-rep file.")
    parser.add_argument("--trace_generate", action="store_true", help="Generate trace file.")
    parser.add_argument("--trace_path", default="./", help="Path to store the trace file. Default: ./")
    parser.add_argument("--device_id", required=True, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--min_n", type=int, default=30, help="Minimum number of samples.")
    parser.add_argument("--print_samples", action="store_true", help="Print the kernels to run. Default: False")
    args = parser.parse_args()

    device_id = args.device_id

    if (args.overwrite or not os.path.exists(f"{args.name}.nsys-rep")):
        nsys_overwrite_flag = "--force-overwrite true" if args.overwrite else ""
        os.system(f"CUDA_VISIBLE_DEVICES={device_id} nsys profile -o {args.name} {nsys_overwrite_flag} {args.cmd}")
        os.system(f"CUDA_VISIBLE_DEVICES={device_id} nsys stats -r cuda_kern_exec_trace -f csv --force-export true {args.name}.nsys-rep > {args.name}.csv")
        os.system(f"CUDA_VISIBLE_DEVICES={device_id} nsys stats -r cuda_gpu_kern_gb_sum -f csv --force-export true {args.name}.nsys-rep > {args.name}_summary.csv")
 
    sample_info_file = parse(args.name, args.threshold, args.error, args.min_n)
    if args.trace_generate: trace_generate(args.cmd, args.trace_path, sample_info_file, device_id)

### Example command
# python3 kernel_sample.py --cmd "python3 olmo-bitnet.py" --name olmo-bitnet --trace_generate --trace_path /data/echung67/trace_sampled/nvbit/olmo-bitnet --device_id 0
