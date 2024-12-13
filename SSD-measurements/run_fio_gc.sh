#!/bin/bash

set -e
if [ "$#" -ne 1 ]; then
    echo "Please specify the NVMe drive to use as /dev/nvmexny"
fi
nvme_drive="$1"
mkdir -p data

# Format the drive
sudo nvme format -s 1 ${nvme_drive}

# Precondition
psrun -d /dev/ttyACM0 -f data/precondition.txt -- \
    sudo ./fio/fio \
    	--name "precondition_fill_nvme" \
    	--filename=${nvme_drive} \
    	--size=100% \
    	--bs=128K \
    	--direct=1 \
    	--rw=write \
    	--thread=1 \
    	--ioengine=io_uring \
    	--fixedbufs=1 \
        --registerfiles=1 \
        --hipri \
    	--write_bw_log=data/precondition \
        --log_avg_msec=1000

# Steady state
psrun -d /dev/ttyACM0 -f data/steadylower.txt -- \
    sudo ./fio/fio \
    	--name "steady_rand_nvme" \
    	--filename=${nvme_drive} \
    	--size=100% \
    	--loops=2 \
    	--bs=4k \
        --time_based=1 \
        --runtime=$((25 * 60))s \
    	--iodepth=1 \
    	--direct=1 \
    	--rw=randwrite \
    	--thread=1 \
    	--ioengine=io_uring \
    	--fixedbufs=1 \
        --registerfiles=1 \
        --hipri \
    	--write_bw_log=steadylower \
        --log_avg_msec=1000;