#!/usr/bin/env python
import datetime
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl

mpl.rcParams.update({'font.size': 16})

import seaborn as sns

sns.set_style("darkgrid")

import numpy as np
import json
import argparse

axlabels = {"TFLOPS": "Compute performance (TFLOP/s)", "TFLOPS/J": "Energy efficiency (TFLOP/J)"}
clocks = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Filename of the Kernel Tuner cache file")
    parser.add_argument("--title", required=True, help="Title of the plot")
    parser.add_argument("--output", required=False, help="Filename of the output file")
    parser.add_argument("--pareto", required=False, action="store_true", help="Include Pareto front")
    return parser.parse_args()

def read_data(filename=None):

    if not filename:
        filename = "AD4000F2_wmma_complex_gemm_opt_16bit_1x4096x4096x4096.json"

    with open(filename, 'r') as fh:
        data = json.loads(fh.read())

    all_data = list(data["cache"].values())

    print(f"Read {len(all_data)} items")

    clocks.extend(data['tune_params']['nvml_gr_clock'])

    # compute how long the experiment took and print
    start_time = datetime.datetime.fromisoformat(all_data[0]['timestamp'])
    end_time = datetime.datetime.fromisoformat(all_data[-1]['timestamp'])
    total_time = end_time - start_time
    print(f"Collecting this data on the GPU took {total_time.total_seconds()} seconds.")

    return all_data


def get_plot():

    f, ax = plt.subplots(figsize=(15, 7))

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.tick_params(axis="both", which="both", bottom=True, top=False,
                labelbottom=True, left=True, right=False, labelleft=True)

    return f, ax


def scatter_plot(metric1, metric2, color_by, title=None, data=None, pareto_front=None, output_file=None):
    f, ax = get_plot()
    scatter(f, ax, metric1, metric2, color_by, title=title, data=data, pareto_front=pareto_front, output_file=output_file)


def scatter(f, ax, metric1, metric2, color_by, title=None, data=None, pareto_front=None, output_file=None):

    if not data:
        data = all_data

    s = ( mpl.rcParams['lines.markersize']**2 ) / 2


    ax.set_xlabel(axlabels[metric2] if metric2 in axlabels else metric2)
    ax.set_ylabel(axlabels[metric1] if metric1 in axlabels else metric1)
    if title:
        ax.set_title(title)

    print(data[0])
    colors = [d[color_by] for d in data if metric1 in d]

    x = [d[metric2] for d in data if metric1 in d]
    y = [d[metric1] for d in data if metric1 in d]

    dots = ax.scatter(x, y, c=colors, cmap=cm.viridis, s=s, alpha=0.5, linewidth=0.1, edgecolor="#757575", label="measurements")

    handles, labels = dots.legend_elements(prop="colors", num=clocks)
    labels = [label + " MHz" for label in labels]

    if pareto_front:
        colors = [d[color_by] for d in pareto_front]
        x = [d[metric2] for d in pareto_front]
        y = [d[metric1] for d in pareto_front]

        ax.scatter(x, y, c=colors, cmap=cm.viridis, s=s, alpha=1, linewidth=0.7, edgecolor="#757575", zorder=2.5)
        pareto_line, = ax.plot(x, y, '-', color="gray", linewidth=0.7, label="pareto front")

        handles.append(pareto_line)
        labels.append("Pareto front")

    ax.legend(handles, labels)

    f.tight_layout()
    if output_file:
        f.savefig(output_file)

    plt.show()


#https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto(costs, maximise=False):
    """
    :param costs: An (n_points, n_costs) array
    :maximise: boolean. True for maximising, False for minimising
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            if maximise:
                is_efficient[is_efficient] = np.any(costs[is_efficient]>=c, axis=1)  # Remove dominated points
            else:
                is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)  # Remove dominated points

    return is_efficient


def find_pareto_front(data=None, metric1="TFLOPS", metric2="TFLOPS/J"):

    if not data:
        data=all_data

    valid_data = [d for d in data if metric1 in d]

    perf = [d[metric1] for d in valid_data]
    energy = [d[metric2] for d in valid_data]

    costs = np.array([perf,energy]).T
    is_efficient = is_pareto(costs, maximise=True)

    np_data = np.array(valid_data)
    pareto_front = np_data[is_efficient]

    #sort on metric2 because that's the x-axis in plots
    pareto_front = sorted(pareto_front, key=lambda x:x[metric2])

    return pareto_front


def print_pareto_front(pareto_front, keys=None):
    def to_string(val):
        if val == int(val):
            return str(val)
        else:
            return f'{val:.4}'
    if not keys:
        keys = pareto_front[0].keys()
    print(" & ".join([k for k in keys]) + " \\\\")
    for d in pareto_front:
        print(" & ".join([to_string(d[k]) for k in keys]) + " \\\\")



if __name__ == "__main__":
    args = parse_args()

    all_data = read_data(args.cache)

    # Compute pareto front
    pareto_front = None
    if args.pareto:
        pareto_front = find_pareto_front(all_data)

    metric1 = "TFLOPS"
    metric2 = "TFLOPS/J"
    color_by = "nvml_gr_clock"
    scatter_plot(metric1, metric2, color_by, title=args.title, data=all_data, pareto_front=pareto_front, output_file=args.output)

    if args.pareto:
        # Print values on the pareto front
        keys = ['nvml_gr_clock', 'block_size_x', 'block_size_y', 'block_size_z', 'M_PER_BLOCK', 'N_PER_BLOCK', 'NBUFFER', 'N_PER_WARP', 'M_PER_WARP', 'TFLOPS', 'TFLOPS/J']
        print_pareto_front(pareto_front, keys)

