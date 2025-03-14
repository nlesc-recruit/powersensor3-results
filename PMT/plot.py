from io import StringIO
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def load_pmt(fname):
    with open(fname) as fh:
        raw = fh.readlines()
    pmt_raw = "".join([line for line in raw if not line.startswith("M")])
    pmt_markers_raw = "".join([line for line in raw if line.startswith("M")])
    pmt_data = np.genfromtxt(StringIO(pmt_raw), names=True, dtype=None)
    pmt_markers = np.genfromtxt(StringIO(pmt_markers_raw), dtype=None)
    return pmt_data, pmt_markers

if __name__ == "__main__":
    folder = sys.argv[1]
    assert folder in ('ad4000', 'w7700')

    ps3 = np.genfromtxt(f"{folder}/pmt-powersensor3.log", names=True, dtype=None)
    if folder == 'ad4000':
        pmt_data, pmt_markers = load_pmt(f"{folder}/pmt-nvml.log")
    else:
        pmt_data, pmt_markers = load_pmt(f"{folder}/pmt-rocm.log")
        pmt_data2, pmt_markers2 = load_pmt(f"{folder}/pmt-amdsmi.log")

    offset = min(ps3["time"][0], pmt_data["timestamp"][0])
    if folder == 'ad4000':
        offset += 9.3
    else:
        offset += 16.1
        # TEMP: scale PS3 results
        ps3["power_total"] *= 150 / 175
    ps3["time"] -= offset
    pmt_data["timestamp"] -= offset
    try:
        pmt_data2["timestamp"] -= offset
    except:
        pass
        
    #print(ps3["time"][0] - pmt_data["timestamp"][0])
        
    fig, ax = plt.subplots()
    ax.plot(ps3["time"], ps3["power_total"], label="PowerSensor3")
    if folder == 'ad4000':
        ax.plot(pmt_data["timestamp"], pmt_data["gpu_instant"], label="NVML instant")
        ax.plot(pmt_data["timestamp"], pmt_data["gpu_average"], label="NVML average")
    else:
        ax.plot(pmt_data["timestamp"], pmt_data["device"], label="ROCm SMI")
        #ax.plot(pmt_data2["timestamp"], pmt_data2["device"], label="AMD SMI")
    for line in ps3:
        if line["marker"] != 'S':
            ax.axvline(line["time"], ls='--', c='b')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    if folder == 'ad4000':
        ax.set_xlim(0, 5.9)
    else:
        ax.set_xlim(0, 4.8)

    plt.savefig(f"PMT_{folder.upper()}.pdf")
    #plt.show()
