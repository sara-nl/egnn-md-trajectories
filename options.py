import os
import argparse
import torch
import time
import random

OPTIMIZERS = ['adam', 'sgd', 'rmsprop']
MOLECULES = ['aspirin', 'benzene_old', 'ethanol', 'malonaldehyde', 'naphthalene', 'salicylic', 'toluene', 'uracil']
TASKS = ['energy', 'forces', 'both']
MOLECULE_2_NUM_ATOMS = {'aspirin': 21,
                        'benzene_old': 12,
                        'ethanol': 9,
                        'malonaldehyde': 9,
                        'naphthalene': 18,
                        'salicylic': 16,
                        'toluene': 15,
                        'uracil': 12}

NUM_INPUT_FEATURES = {'aspirin': 9,
                      'benzene_old': 6,
                      'ethanol': 9,
                      'malonaldehyde': 9,
                      'naphthalene': 6,
                      'salicylic': 9,
                      'toluene': 6,
                      'uracil': 12}


def get_options():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    parser.add_argument('--molecule', type=str, default='aspirin', choices=MOLECULES, help='Which molecule to use')

    # Basic training options
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs during training')
    parser.add_argument('--energy', action='store_true', help='Train on energy only')
    parser.add_argument('--force_and_energy', action='store_true', help='Train of the forces and energy')
    parser.add_argument('--force', action='store_true', help='Train on force only')
    parser.add_argument('--p', type=int, default=20, help='Force loss factor')

    # Network options
    parser.add_argument('--hidden_nf', type=int, default=128, help='Number of hidden features')
    parser.add_argument('--n_layers', type=int, default=7, help='Number of layers')

    # Optimizer, learning rate (scheduler) and early stopping options
    parser.add_argument('--optimizer', type=str, choices=OPTIMIZERS, default='adam')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for Adam optimizer')
    parser.add_argument('--sgd_momentum', type=float, default=0.9, help='Momentum term for SGD')
    parser.add_argument('--sgd_nesterov', action='store_true', help='Use nesterov moment with SGD')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=1e-16, help='Weight decay for gradient updates')
    parser.add_argument('--scheduler', type=str, choices=['cosine'], default='cosine')

    # Cuda options
    parser.add_argument('--no_cuda', action='store_true', help='Use this to train without cuda enabled')
    parser.add_argument('--cuda_devices', nargs='+', default=[0])
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')

    # Logging options
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output files to')

    # Inference options
    parser.add_argument('--model_file', type=str, default=None, help='Name of pretrained model')

    # Misc
    parser.add_argument('--seed', type=int, default=None, help='Random seed to use')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 4, help='Number of workers for dataloader')

    opts = parser.parse_args()

    assert opts.energy or opts.force_and_energy or opts.force, "Must train on energies, forces or both"

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.cuda_devices = list(sorted([int(i) for i in opts.cuda_devices]))
    opts.device = f"cuda:{opts.cuda_devices[0]}" if opts.use_cuda else "cpu"

    if opts.seed is None:
        opts.seed = opts.cuda_devices[0]

    opts.dataset_path = 'data/' + opts.molecule + '_dft.npz'
    opts.num_atoms = MOLECULE_2_NUM_ATOMS[opts.molecule]
    opts.num_input_feat = NUM_INPUT_FEATURES[opts.molecule]

    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    if opts.energy:
        opts.output_dir = os.path.join(opts.output_dir, 'energy', opts.molecule, opts.run_name)
    elif opts.force:
        opts.output_dir = os.path.join(opts.output_dir, 'force', opts.molecule, opts.run_name)
    else:
        opts.output_dir = os.path.join(opts.output_dir, 'force_and_energy', opts.molecule, opts.run_name)

    os.makedirs(opts.output_dir, exist_ok=True)

    return opts
