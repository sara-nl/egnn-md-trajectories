import random
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter
import argparse
import json
from pprint import pprint

from train import train_epoch, evaluate_epoch
from data.molecule_dataset import get_dataset, collate_fn
# from model.original_egnn import EGNN, TwoHeadedEGNN, ForceEGNN
from model.clean_egnn import EGNN
from options import get_options
from utils import get_num_parameters


def main(opts: argparse.Namespace) -> None:
    """
    Main function to run
    :param opts: the argparse object
    :return: None
    """
    pprint(vars(opts))

    with open(os.path.join(opts.output_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the random seed
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    np.random.seed(opts.seed)

    model = EGNN(in_node_nf=opts.num_input_feat, in_edge_nf=1, hidden_nf=opts.hidden_nf,
                     n_layers=opts.n_layers, attention=True, num_atoms=opts.num_atoms)
    print(f'The model has {get_num_parameters(model)} parameters')

    if opts.model_file is not None:
        state_dict = torch.load(opts.model_file, map_location='cpu')
        model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1 and len(opts.cuda_devices) > 1:
        model = nn.DataParallel(model, device_ids=opts.cuda_devices)
    model = model.to(opts.device)

    if opts.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.999), eps=opts.eps,
                               weight_decay=opts.weight_decay, amsgrad=False)
    elif opts.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=opts.lr,
                                  weight_decay=opts.weight_decay, momentum=opts.sgd_momentum)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=opts.sgd_momentum,
                              nesterov=opts.sgd_nesterov,
                              weight_decay=opts.weight_decay)

    if opts.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opts.epochs)
        opts.scheduler_update = 'epoch'
    else:
        scheduler = None
        opts.scheduler_update = None

    train_dataset = get_dataset(opts, train=True, seed=opts.seed)
    valid_dataset = get_dataset(opts, train=False, seed=opts.seed)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
                              shuffle=True, collate_fn=collate_fn, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.eval_batch_size, num_workers=opts.num_workers,
                              shuffle=False, collate_fn=collate_fn, drop_last=False)
    criterion = torch.nn.L1Loss()

    # Define the training and validation functions
    train_func = train_epoch
    val_func = evaluate_epoch

    # Define the GradScaler object
    scaler = GradScaler(enabled=opts.mixed_precision)

    tb_logger = SummaryWriter(opts.output_dir, flush_secs=5)

    best_test_loss_epoch = float('inf')

    for epoch in range(opts.epochs):

        train_output = train_func(model, criterion, optimizer, train_loader, opts, scaler=scaler,
                                  p=opts.p, scheduler=scheduler)

        val_output = val_func(model, criterion, valid_loader, opts, p=opts.p)

        #########################
        #    Write to logger    #
        #########################

        tb_logger.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        tb_logger.add_scalar('Loss/Train', train_output['train_loss_epoch'], epoch)
        tb_logger.add_scalar('Loss/Val', val_output['test_loss_epoch'], epoch)

        tb_logger.add_scalar('Loss_Energy/Train', train_output['energy_loss_epoch'], epoch)
        tb_logger.add_scalar('Loss_Energy/Val', val_output['energy_loss_epoch'], epoch)

        tb_logger.add_scalar('Loss_Force_grad/Train', train_output['force_loss_epoch'], epoch)
        tb_logger.add_scalar('Loss_Force_grad/Val', val_output['force_loss_epoch'], epoch)

        tb_logger.add_scalar('MAE_energy/Train', train_output['mae_energy'], epoch)
        tb_logger.add_scalar('MAE_energy/Val', val_output['mae_energy'], epoch)

        tb_logger.add_scalar('MAE_forces_grad/Train', train_output['mae_force'], epoch)
        tb_logger.add_scalar('MAE_forces_grad/Val', val_output['mae_force'], epoch)

        #########################
        #          Misc         #
        #########################

        if val_output['test_loss_epoch'] < best_test_loss_epoch:
            best_test_loss_epoch = val_output['test_loss_epoch']
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(),
                           os.path.join(opts.output_dir, f'best_{opts.molecule}_state_dict.pth'))
            else:
                torch.save(model.state_dict(),
                           os.path.join(opts.output_dir, f'best_{opts.molecule}_state_dict.pth'))

        if opts.scheduler is not None and opts.scheduler_update == 'epoch':
            scheduler.step()


if __name__ == '__main__':
    opts = get_options()
    main(opts)
