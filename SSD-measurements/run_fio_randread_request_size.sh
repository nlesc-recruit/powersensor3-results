#!/bin/bash

set -e
if [ "$#" -ne 1 ]; then
    echo "Please specify the NVMe drive to use as /dev/nvmexny"
fi
nvme_drive="$1"
mkdir -p data

for bs in $(seq 1 4096); do 
    psrun -d /dev/ttyACM0 -f data/read_bs_${bs}k.txt -- \
        sudo ./fio/fio \
            --rw=randread \
            --time_based=1 \
            --runtime=$((1 * 10))s \
            --ioengine=io_uring \
            --registerfiles=1 \
            --hipri=1 \
            --fixedbufs=1 \
            --thread=1 \
            --direct=1 \
            --sqthread_poll=1 \
            --numjobs=1 \
            --iodepth=1 \
            --bs=${bs}k \
            --filename=${nvme_drive} \
            --name=read \
            --write_bw_log=read_bs_${bs}k \
            --log_avg_msec=1000; 
done
