# Using PowerSensor3 with Kernel Tuner

## Requirements

To rerun the data collection, the following is required:

1. The PowerSensor3 hardware installed in your system
2. The PowerSensor3 host library compiled with Python interface
3. A GPU (either Nvidia or AMD) with a CUDA or HIP installation
4. Python 3
5. Kernel Tuner with either cupy or hip

To reproduce the analysis and plots from the paper from the (pre-)collected data, the following is required:

1. Python 3
2. pip install -r requirements.txt

## How to reproduce

The performance data used in the paper is already contained in this repository. To re-run the analysis and reproduce the plots, there is no need to re-collect the 
performance measurement data.

### Re-collecting the data

However, to fully reproduce the results from the paper, re-collecting the performance measurement data using PowerSensor3 and Kernel Tuner can be done using the following command:
```bash
python tune_gemm.py --name AD4000 --nbit 16 -b 1 -m 4096 -n 4096 -k 4096 --backend cupy --observer powersensor --ccglib /home/bwn200/ccglib --freq "1485, 1515, 1560, 1590, 1635, 1665, 1710, 1740, 1785, 1815"
```
This scipt requires the PowerSensor3 hardware to be installed into the machine.
This also requires a GPU with root privileges to control clock frequencies.
Collecting this data takes several hours and produces a file named AD4000_wmma_complex_gemm_opt_16bit_1x4096x4096x4096.json. This file is already part of this repository.

For a Jetson AGX Orin, the procedure is similar:
```
python tune_gemm.py --name Orin --nbit 16 -b 1 -m 4096 -n 4096 -k 4096 --backend cupy --observer powersensor --ccglib /home/bwn200/ccglib --freq 0 --tegra
```
The `--freq 0` settings automatically runs over all available frequencies. The resulting file Orin_wmma_complex_gemm_opt_16bit_1x4096x4096x4096.json is also available in this repository.

### Running the analysis and reproducing the plots

The tune_plot.py script can be used to process the collected performance data. It also produces the plots in the paper, for example using:
```bash
python tune_plot.py --cache AD4000_wmma_complex_gemm_opt_16bit_1x4096x4096x4096.json --title "Tensor-Core Beamformer 16-bit 4kx4kx4k on RTX 4000 Ada" --output beamformer_ad4000.pdf --pareto
python tune_plot.py --cache Orin_wmma_complex_gemm_opt_16bit_1x4096x4096x4096.json --title "Tensor-Core Beamformer 16-bit 4kx4kx4k on Jetson AGX Orin" --output beamformer_orin.pdf --pareto
```
The --pareto switch can be used to highlight the pareto front in the plot. It also prints the Latex code for the table.

