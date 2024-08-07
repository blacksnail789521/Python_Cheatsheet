import torch
import argparse
from pathlib import Path

from exp.exp import Exp_Classification
from utils.visual import plot_predictions
from utils.tools import set_seed, print_formatted_dict, select_best_metrics


def get_args_from_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TimeDRL")

    # * basic config
    # parser.add_argument(
    #     "--task_name",
    #     type=str,
    #     default="forecasting",
    #     choices=["forecasting", "classification"],
    #     help="time-series tasks",
    # )
    parser.add_argument(
        "--overwrite_args",
        action="store_true",
        help="overwrite args with fixed_params and tunable_params",
        default=False,
        # ! Remeber to run `python main.py --overwrite_args` to overwrite args
    )
    parser.add_argument(
        "--checkpoint_loading_path",
        type=str,
        default="./checkpoints/checkpoint.pth",
        help="checkpoint loading path",
    )
    parser.add_argument(
        "--checkpoint_saving_path",
        type=str,
        default="./checkpoints/checkpoint.pth",
        help="checkpoint saving path",
    )
    parser.add_argument(
        "--use_tqdm",
        action="store_true",
        help="use tqdm for progress bar",
        # default=False,
        default=True,
    )

    parser.add_argument("--n_layers", type=int, default=12, help="number of layers")
    parser.add_argument("--n_heads", type=int, default=12, help="number of heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        help="activation function",
        choices=["relu", "gelu"],
    )
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--enable_channel_independence",
        type=bool,
        default=True,
        help="enable channel independence",
    )

    # * model architecture
    parser.add_argument(
        "--model_name",
        type=str,
        default="MLP",
        choices=["MLP", "CNN"],
        help="model name",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="number of layers",
    )
    parser.add_argument(
        "--use_bn",
        type=bool,
        default=True,
        help="use batch normalization",
    )

    # * training_stage_params
    parser.add_argument(
        "--optim",
        type=str,
        default="AdamW",
        help="optimizer",
        choices=["Adam", "AdamW", "RMSprop"],
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="StepLR",
        help="learning rate scheduler",
        choices=[
            "none",
            "StepLR",  # step_size=30, gamma=0.1. Decays the LR by gamma every step_size epochs.
            "ExponentialLR",  # gamma=0.95
            "ReduceLROnPlateau",  # factor=0.1, patience=10
            "CosineAnnealingLR",  # T_max=50
            "CyclicLR",  # base_lr=lr, max_lr=0.1, step_size_up=20
            "OneCycleLR",  # max_lr=0.1, steps_per_epoch=len(train_loader), epochs=num_epochs
        ],
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="gamma for lr_scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="l2 weight decay",
    )
    parser.add_argument("--epochs", type=int, default=10, help="epochs for training")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="data loader num workers"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="early stopping patience"
    )
    parser.add_argument("--delta", type=float, default=0, help="early stopping delta")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        # default=False,
        default=True,  # faster
    )

    # * GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")

    args, _ = parser.parse_known_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.root_folder = Path.cwd()  # Set this outside of the trainable function
    args.lr_scheduler_params = {}

    return args


def update_args_from_fixed_params(
    args: argparse.Namespace, fixed_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in fixed_params.items():
        print("### [Fixed] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args_from_tunable_params(
    args: argparse.Namespace, tunable_params: dict
) -> argparse.Namespace:
    # Update args
    for key, value in tunable_params.items():
        print("### [Tunable] Set {} to {}".format(key, value))
        setattr(args, key, value)

    return args


def update_args(
    args: argparse.Namespace,
    fixed_params: dict,
    tunable_params: dict,
) -> argparse.Namespace:
    # Check if there are duplicated keys
    duplicated_keys = set(fixed_params.keys()) & set(tunable_params.keys())
    assert not duplicated_keys, f"Duplicated keys found: {duplicated_keys}"

    # Update args from fixed_params, tunable_params, and dataset
    if args.overwrite_args:
        args = update_args_from_fixed_params(args, fixed_params)
        args = update_args_from_tunable_params(args, tunable_params)
    # args = update_args_from_dataset(args)

    print(f"Args in experiment: {args}")

    return args


def trainable(
    tunable_params: dict,
    fixed_params: dict,
    args: argparse.Namespace,
    enable_plot: bool = False,
) -> dict:
    # Update args
    args = update_args(args, fixed_params, tunable_params)

    # Train the model
    exp = Exp_Classification(args)
    metrics = exp.train()
    print_formatted_dict(metrics)

    # Plot predicitons
    if enable_plot:
        print("---------------------------------------")
        print("Plotting predictions ...")
        plot_predictions(
            exp.model, exp.test_loader, fixed_params["root_path"], enable_show=True
        )

    return select_best_metrics(metrics, target_mode="test", target_metric="acc")


if __name__ == "__main__":
    """------------------------------------"""
    model_name = "MLP"
    # model_name = "CNN"

    batch_size = 256

    num_workers = 4
    """------------------------------------"""
    # Set all random seeds (Python, NumPy, PyTorch)
    set_seed(42)

    # Setup args
    args = get_args_from_parser()

    # Setup fixed params
    fixed_params = {
        "root_path": Path.cwd(),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    # Setup tunable params
    tunable_params = {
        "model_name": model_name,
        "model_params": {
            "MLP": {
                "num_layers": 3,
                "use_bn": True,
            },
            "CNN": {
                "num_conv_layers": 3,
                "use_bn": True,
            },
        },
        "optim": "Adam",
        # "optim": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        # "epochs": 1,
        "epochs": 3,
        # "lr_scheduler": "none",
        "lr_scheduler": "StepLR",
        # "lr_scheduler": "ReduceLROnPlateau",
        "lr_scheduler_params": {
            "StepLR": {
                "step_size": 1,
                "gamma": 0.1,
            },
            "ExponentialLR": {
                "gamma": 0.95,
            },
            "ReduceLROnPlateau": {
                "factor": 0.1,
                "patience": 10,
            },
            "CosineAnnealingLR": {
                "T_max": 2,
            },
            "CyclicLR": {
                "max_lr": 0.1,
                "step_size_up": 3,
                "step_size_down": 3,
            },
            "OneCycleLR": {
                "max_lr": 0.1,
            },
        },
    }

    # Run
    best_metrics = trainable(tunable_params, fixed_params, args, enable_plot=True)
    print_formatted_dict(best_metrics)
    print("### Done ###")
