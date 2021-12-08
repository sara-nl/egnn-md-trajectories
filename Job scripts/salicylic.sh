#!/bin/bash
#Set job requirements
#SBATCH -p gpu_titanrtx
#SBATCH -t 5-00:00:00
#SBATCH -N 1
#SBATCH -J salicylic

cd ..
source init.sh

python3 main.py --force_and_energy --lr 5e-5 --batch_size 96 --molecule salicylic --p 1 --output_dir outputs/1 --run_name salicylic_1 --cuda_devices 0 &
python3 main.py --force_and_energy --lr 5e-5 --batch_size 96 --molecule salicylic --p 2 --output_dir outputs/2 --run_name salicylic_2 --cuda_devices 1 &
python3 main.py --force_and_energy --lr 5e-5 --batch_size 96 --molecule salicylic --p 5 --output_dir outputs/5 --run_name salicylic_5 --cuda_devices 2 &
python3 main.py --force_and_energy --lr 5e-5 --batch_size 96 --molecule salicylic --p 10 --output_dir outputs/10 --run_name salicylic_10 --cuda_devices 3
