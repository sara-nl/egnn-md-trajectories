from typing import NoReturn, Union, Tuple
import torch


def print_epoch_progress(train_loss: float, val_loss: float, train_mae: Union[float, Tuple[float, float]],
                         val_mae: Union[float, Tuple[float, float]], train_force_mae: float, test_force_mae: float,
                         epoch: int = None, n_epoch: int = None) -> NoReturn:
    """Print all the information after each epoch.
        :train_loss: average training loss
        :val_loss: average validation loss
        :train_mae: training mean absolute error
        :train_mae: validation mean absolute error
        :train_force_mae training mean absolute error of the force prediction
        :test_force_mae validation mean absolute error of the force prediction
        :epoch: current epoch
        :n_epoch: the total number of epochs
        :task: the task, should be 'energy', 'forces', or 'both'
        :returns: None
    """

    log_str = ''
    if epoch is not None and n_epoch is not None:
        log_str += 'Ep: {0}/{1}|'.format(epoch + 1, n_epoch)

    log_str += 'Train/Val loss: {:.4f}/{:.4f}|'.format(train_loss, val_loss)

    log_str += 'MAE energy: {:.4f}/{:.4f}|'.format(train_mae, val_mae)

    log_str += 'MAE forces: {:.4f}/{:.4f}|'.format(train_force_mae, test_force_mae)

    print(log_str)


def get_num_parameters(model: torch.nn.Module) -> int:
    """
    Get the number of parameters of the model

    :param model: the model
    :return: the number of parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
