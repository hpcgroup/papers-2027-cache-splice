#!/bin/bash
# Regenerate every data-driven figure of the paper with the PSSG template (../pssg-plots, or $PSSG_PLOTS).
# model/make_data.py needs the gt3d checkout (../gt3d, or $GT3D); the other scripts only read the CSV files.
set -e
cd "$(dirname "$0")"
source /pscratch/sd/c/cunyang/gnn/plexus_env/bin/activate
python -W ignore model/make_data.py
python -W ignore breakdown/make_data.py
python -W ignore samebatch/make_data.py
python -W ignore kernel/make_data.py
for d in torchgt strategies kernel model scaling sparsecomm breakdown epoch_time scaling_best samebatch; do
    (cd "$d" && python -W ignore plot.py)
done
