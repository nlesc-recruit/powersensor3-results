#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
import os as os
import datetime
import seaborn as sns

sns.set_style('darkgrid')
sensor_labels = ['3.3V 10A', '12V 10A', 'Ext 12V 20A', 'USB-C 20V 5A']
line_color = ['b', 'r', 'k', 'k']

def read_datafile(file_name, sensor):
    with open(file_name, newline='') as csvfile:
        data = []
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            data.append(row)
    data = np.array(data[1:])
    volt = [float(dat) for dat in data.transpose()[4+1+sensor*3].flatten()]
    current = [float(dat) for dat in data.transpose()[4+sensor*3].flatten()]
    power = [volt[cnt]*current[cnt] for cnt in range(len(volt))]
    time = [float(dat) for dat in data.transpose()[1].flatten()]
    dtime = [float(dat) for dat in data.transpose()[2].flatten()]
    return volt, current, power, time, dtime


def plot_step(power):
    plt.figure()
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['figure.figsize'] = [15, 7]
    fig, (ax1, ax2) = plt.subplots(1, 2)

    start_point_1 = 35
    points = 150
    time_ms = [start_point_1*50e-3 + cnt * 50e-3 for cnt in range(points)]
    for power in powers:
        ax1.plot(time_ms, power[start_point_1:start_point_1+points], label="Max.")
    ax1.set(xlabel=f"Time (ms)" , ylabel="Power (W)")
    ax1.set_position([0, 0, 0.50, 0.9])
    ax1.set_ylim(ymin=0, ymax=15)
    ax1.set_xlim(2, 9)

    start_point_2 = start_point_1+15
    points = 22
    time_ms = [start_point_2*50 + cnt * 50 for cnt in range(points)]
    for power in powers:
        ax2.plot(time_ms, power[start_point_2:start_point_2+points], label="Max.")
    ax2.set(xlabel=f"Time ($\mu $s)" , ylabel="Power (W)")
    ax2.grid(True)
    ax2.set_position([0.6, 0, 0.25, 0.90])
    ax2.set_ylim(ymin=0, ymax=15)
    ax2.set_xlim(2500, 3500)
    fig.suptitle("PowerSensor3 step response", y=1)
    fig.savefig("step.pdf", bbox_inches = 'tight')
    fig.show()


def main():
    file_location = './step_data'
    sensors = [2]
    process_files = ['step_100Hz_90pr_1A.csv']
    voltages=[]
    currents=[]
    cur_max=[]
    cur_min=[]
    volt_max=[]
    volt_min=[]
    time_scale=[]
    powers=[]
    powers_mean =[]
    pwr_max=[]
    pwr_min=[]
    for file_name in process_files:
        current=[]
        file_name = file_location + file_name
        file_time = os.path.getmtime(file_name)
        time_scale.append(os.path.getmtime(file_name))
        for sensor in sensors:
            volt, current, power, time, dtime = read_datafile(file_name, sensor)
            voltages.append(np.mean(volt))
            volt_max.append(np.max(volt))
            volt_min.append(np.min(volt))
            currents.append(np.mean(current))
            cur_max.append(np.max(current))
            cur_min.append(np.min(current))
            powers.append(power)
            powers_mean.append(np.mean(power))
            pwr_max.append(np.max(power))
            pwr_min.append(np.min(power))
    plot_step(power)
