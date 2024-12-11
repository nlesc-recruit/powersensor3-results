# Using PowerSensor3 with fio

## Requirements

To rerun the data collection, the following is required:

1. The PowerSensor3 hardware installed in your system
2. The PowerSensor3 host library installed (Python interface not necessary)
3. An NVMe SSD that is attached to the PowerSensor3
4. fio (see submodule)

To install the benchmark dependencies, the following is required:

1. ./build_fio.sh
2. ./fio/fio --version (should return fio-3.38-15-gdc03)

To reproduce the analysis and plots from the paper, the following is required:

1. Python 3
2. pip install -r requirements.txt
3. jupyter --version

## How to reproduce

The data used in the paper is not (yet) present in this repository. To retrieve the data and reproduce the plots, the fio scripts will have to be run. The data used in the paper will be published on Zenodo soon.

## Re-collecting the data

> **WARNING**: The fio experiments wipe ALL data present on the NVMe drive. Do not run it on an SSD that has important data.

To reproduce the paper results from both SSD experiments, the following commands should be used:

```bash
./run_fio_randread_request_size.sh <NVME DRIVE TO WIPE>
./run_fio_gc.sh <NVME DRIVE TO WIPE>
```

## Running the analysis and reproducing the plots

The following notebook includes the plots needed to reproduce the plots in the paper:

```bash
jupyter notebook
# Open plots.ipynb and run
```
