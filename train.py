from tqdm import tqdm
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.autograd import grad
import torch
import argparse
import numpy as np
import pdb


def move_to(var, device):
    if var is None:
        return None
    elif isinstance(var, (int, str, float)):
        return var
    elif isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(k, device) for k in var]
    elif isinstance(var, tuple):
        return (move_to(k, device) for k in var)
    return var.to(device)


def train_epoch(model: torch.nn.Module,
                criterion: torch.nn.modules.loss._Loss,
                optimizer: torch.optim.Optimizer,
                train_loader: torch.utils.data.DataLoader,
                opts: argparse.Namespace,
                p: int = 20,
                scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                scaler: torch.cuda.amp.GradScaler = None) -> dict:
    """
    Train on the energy task for one epoch

    :param model: the model
    :param criterion: the loss function
    :param optimizer: the optimizer
    :param train_loader: the train data loader
    :param opts: the options object
    :param p: the scaling of the force loss term
    :param scheduler: the learning rate scheduler
    :param scaler: the mixed precision scaler
    :return: train_loss_epoch and mae
    """
    # Put model in train mode and reset gradients
    model.train()
    model.zero_grad()

    train_losses, energy_losses, force_losses, energy_errors, force_errors = [], [], [], [], []

    for batch_id, batch in enumerate(tqdm(train_loader)):

        batch_size = batch['batch_size']
        batch = move_to(batch, opts.device)

        optimizer.zero_grad()

        with autocast(enabled=opts.mixed_precision):

            # Set to require gradients
            if opts.force_and_energy:
                batch['coords'].requires_grad_(True)

            prediction = model(h=batch['nodes'],
                               x=batch['coords'],
                               edges=batch['edges'],
                               edge_attr=batch['edge_attr'])

            if opts.force_and_energy:
                force = -grad(outputs=prediction['energy'], inputs=batch['coords'],
                              grad_outputs=torch.ones_like(prediction['energy']), create_graph=True, retain_graph=True)[
                    0]
                force = force.view(batch_size, -1, force.shape[-1])
                e_loss = criterion(prediction['energy'], (batch['energies'] - batch['energy_meann']) / batch['energy_mad'])
                f_loss = p * criterion(force, (batch['forces'] - batch['force_meann'][0]) / batch['force_mad'][0])
                loss = e_loss + f_loss
                f_pred_error = criterion((force * batch['force_mad'][0]) + batch['force_meann'][0], batch['forces'])
            else:
                e_loss = criterion(prediction['energy'],
                                   (batch['energies'] - batch['energy_meann']) / batch['energy_mad'])
                f_loss = f_pred_error = torch.tensor([0], device='cpu')
                loss = e_loss

            e_pred_error = criterion((prediction['energy'] * batch['energy_mad']) + batch['energy_meann'],
                                     batch['energies'])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and opts.scheduler_update == 'batch':
            scheduler.step()

        train_losses.append(loss.detach().cpu().item())
        energy_losses.append(e_loss.detach().cpu().item())
        force_losses.append(f_loss.detach().cpu().item())

        energy_errors.append(e_pred_error.detach().cpu().item())
        force_errors.append(f_pred_error.detach().cpu().item())

    train_loss_epoch = np.round(np.mean(train_losses), 4)
    energy_loss_epoch = np.round(np.mean(energy_losses), 4)
    force_loss_epoch = np.round(np.mean(force_losses), 4)
    energy_errors = np.round(np.mean(energy_errors), 4)
    force_errors = np.round(np.mean(force_errors), 4)

    # Convert from kcal/mol to meV
    mae_energy = energy_errors * 0.0433641153087705 * 1000
    mae_force = force_errors * 0.0433641153087705 * 1000

    return {'train_loss_epoch': train_loss_epoch,
            'energy_loss_epoch': energy_loss_epoch,
            'force_loss_epoch': force_loss_epoch,
            'mae_energy': mae_energy,
            'mae_force': mae_force}


def evaluate_epoch(model: torch.nn.Module,
                   criterion: torch.nn.modules.loss._Loss,
                   test_loader: torch.utils.data.DataLoader,
                   opts: argparse.Namespace,
                   p: int = 20) -> dict:
    """
    Evaluate the validation dataset for the energy task
    :param model: the model
    :param criterion: the loss function
    :param test_loader: the test_loader
    :param opts: the options object
    :param p: the scaling of the force loss term
    :return: the test_loss_epoch and mae
    """
    # Put model in eval mode and reset gradients
    model.eval()
    model.zero_grad()

    test_losses, energy_losses, force_losses, energy_errors, force_errors = [], [], [], [], []

    for batch_id, batch in enumerate(tqdm(test_loader)):

        batch_size = batch['batch_size']
        batch = move_to(batch, opts.device)

        with autocast(enabled=opts.mixed_precision):

            # Set to require gradients
            if opts.force_and_energy:
                batch['coords'].requires_grad_(True)

            prediction = model(h=batch['nodes'],
                               x=batch['coords'],
                               edges=batch['edges'],
                               edge_attr=batch['edge_attr'])

            if opts.force_and_energy:
                force = -grad(outputs=prediction['energy'], inputs=batch['coords'],
                              grad_outputs=torch.ones_like(prediction['energy']), create_graph=True, retain_graph=True)[
                    0]
                force = force.view(batch_size, -1, force.shape[-1])
                e_loss = criterion(prediction['energy'], (batch['energies'] - batch['energy_meann']) / batch['energy_mad'])
                f_loss = p * criterion(force, (batch['forces'] - batch['force_meann'][0]) / batch['force_mad'][0])
                loss = e_loss + f_loss
                f_pred_error = criterion((force * batch['force_mad'][0]) + batch['force_meann'][0], batch['forces'])
            else:
                e_loss = criterion(prediction['energy'],
                                   (batch['energies'] - batch['energy_meann']) / batch['energy_mad'])
                f_loss = f_pred_error = torch.tensor([0], device='cpu')
                loss = e_loss

            e_pred_error = criterion((prediction['energy'] * batch['energy_mad']) + batch['energy_meann'],
                                     batch['energies'])

        test_losses.append(loss.detach().cpu().item())
        energy_losses.append(e_loss.detach().cpu().item())
        force_losses.append(f_loss.detach().cpu().item())

        energy_errors.append(e_pred_error.detach().cpu().item())
        force_errors.append(f_pred_error.detach().cpu().item())

    test_loss_epoch = np.round(np.mean(test_losses), 4)
    energy_loss_epoch = np.round(np.mean(energy_losses), 4)
    force_loss_epoch = np.round(np.mean(force_losses), 4)
    energy_errors = np.round(np.mean(energy_errors), 4)
    force_errors = np.round(np.mean(force_errors), 4)

    # Convert from kcal/mol to meV
    mae_energy = energy_errors * 0.0433641153087705 * 1000
    mae_force = force_errors * 0.0433641153087705 * 1000

    return {'test_loss_epoch': test_loss_epoch,
            'energy_loss_epoch': energy_loss_epoch,
            'force_loss_epoch': force_loss_epoch,
            'mae_energy': mae_energy,
            'mae_force': mae_force}
