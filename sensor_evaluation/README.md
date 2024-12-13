# Evaluation the PowerSensor3.

## Requirements

To rerun the data collection, the following is required:

1. The PowerSensor3 hardware installed in your system
2. Power supply
3. Electronic load e.g. Kneil Electronic load
4. PowerSensor3 envirionment including pstest 
5. Python 3

To reproduce the analysis and plots from the paper from the (pre-)collected data, the following is required:

1. Python 3
2. pip install -r requirements.txt

## How to reproduce

The sensor data used in the paper is already contained in this repository. Unzip the data in the .\current_sweep_data to .\current_sweep_data 
The plot_current_sweep.py script can be used to plot the collected sensor data. It also produces the plots in the paper, for example using:
```bash
python plot_current_sweep.py

```

### Re-collecting the data

A detailed description of the measruements can be found here: https://git.astron.nl/RD/powersensor3/-/tree/main/Measurements?ref_type=heads


