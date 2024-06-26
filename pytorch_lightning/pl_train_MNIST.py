import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import netron
import matplotlib.pyplot as plt
from pathlib import Path

# import lightning as L
import pytorch_lightning as L  # 2.0.0

# from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torchmetrics
import numpy as np
from datetime import datetime
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from pl_load_MNIST import load_MNIST, show_data
from pl_load_MNIST_DataModule import MNIST_DataModule
from models.MLP import MLP
from models.CNN import CNN
from models.LightningModuleWrapper import LightningModuleWrapper


def plot_model_with_netron(model: nn.Module, name: str = "DNN") -> None:
    # Save the model
    model_path = Path("saved_models", f"{name}.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path)  # Don't use .state_dict()

    # Plot the model
    netron.start(model_path, address=8081)


def train_model(
    model: L.LightningModule,
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int = 3,
    enable_checkpointing: bool = True,
    enable_logging: bool = True,
    additional_callbacks: list = [],
    use_gpu: bool = False,
    devices: int | list[int] | str = "auto",
    verbose: bool = False,
) -> L.Trainer:
    # Set callbacks
    callbacks = []
    # (We don't need to set the tensorboard logger because it is set by default)
    if enable_checkpointing:
        # We don't want to use the default one because it doesn't save all models
        # We need to save all models because we want to use the best model for the test set
        model_checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch:02d}-{val_loss:.4f}",
            save_top_k=-1,  # save all models
            save_weights_only=True,
        )
        callbacks.append(model_checkpoint)
    early_stopping_with_TerminateOnNaN = EarlyStopping(
        monitor="val_loss", mode="min", patience=3, verbose=verbose
    )
    callbacks.append(early_stopping_with_TerminateOnNaN)
    callbacks.extend(additional_callbacks)

    # Set trainer
    default_root_dir = Path(
        "ray_results",
        f"{model.name}",
        f"train_a_model",
        datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    )
    device_params = {}
    if not use_gpu:
        device_params["accelerator"] = "cpu"
    else:
        device_params["accelerator"] = "gpu"
        device_params["devices"] = devices
        # device_params["strategy"] = "ddp"  # Since 2.0.0, we need to use ddp
        device_params["strategy"] = (
            "ddp_find_unused_parameters_false"  # Allow to have unused parameters
        )
    trainer = L.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=epochs,
        log_every_n_steps=50,  # default: 50
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing,
        logger=enable_logging,
        # We don't need to save the model because we use ModelCheckpoint
        **device_params,
    )

    # Train the model
    trainer.fit(model, train_dl, val_dl)

    # Destroy ddp
    if use_gpu and device_params.get("strategy", None) is not None:
        torch.distributed.destroy_process_group()  # type: ignore

    return trainer


def plot_predictions(
    model: L.LightningModule,
    tester: L.Trainer,
    test_dl: DataLoader,
    log_dir: Path,
    enable_show: bool = True,
) -> None:
    # Get all the predictions (y_pred_list[0].shape: (32, 10))
    y_pred_list = tester.predict(model, test_dl)
    y_pred = y_pred_list[0]  # Extract the first batch  # type: ignore

    # Show the first 5 predictions
    x, y = next(iter(test_dl))
    for i in range(5):
        plt.imshow(x[i, 0, :, :], cmap="gray")
        plt.title(f"Label: {y[i]}, Prediction: {np.argmax(y_pred[i])}")

        # Save the figure
        plot_folder = log_dir / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_folder / f"prediction_{i}.png")
        if enable_show:
            plt.show()
        plt.close()


def get_nn_model(tunable_params: dict, fixed_params: dict) -> nn.Module:
    if fixed_params["model_name"] == "MLP":
        nn_model = MLP(tunable_params["num_layers"])
    elif fixed_params["model_name"] == "CNN":
        nn_model = CNN(tunable_params["num_layers"])
    else:
        raise ValueError(f"Unknown model name: {fixed_params['model_name']}")

    return nn_model


def trainable(
    tunable_params: dict,  # Place tunable parameters first for Ray Tune
    fixed_params: dict,
    ray_tune: bool = True,
    use_lightning_data_module: bool = True,
    data_dir: str = "./",
) -> None:
    # Load data
    if not use_lightning_data_module:
        train_dl, test_dl = load_MNIST(
            batch_size=tunable_params["batch_size"],
            max_concurrent_trials=fixed_params.get("max_concurrent_trials", 1),
        )
        val_dl = test_dl
    else:
        dm = MNIST_DataModule(
            data_dir=data_dir,
            batch_size=tunable_params["batch_size"],
            split=0.8,
            max_concurrent_trials=fixed_params.get("max_concurrent_trials", 1),
        )
        dm.prepare_data()
        dm.setup()
        train_dl = dm.train_dataloader()
        val_dl = dm.val_dataloader()
        test_dl = dm.test_dataloader()
    if not ray_tune:
        show_data(train_dl)  # Show the data

    # Get the nn_model
    nn_model = get_nn_model(tunable_params, fixed_params)

    # Convert to the lightning module
    model = LightningModuleWrapper(
        nn_model=nn_model,
        loss=fixed_params["loss"],
        metrics=fixed_params["metrics"],
        **tunable_params,
    )
    if not ray_tune:
        print(model)

        # Plot the model
        # plot_model_with_netron(model)

    # Determine additional_callbacks (for logging/plotting purposes only)
    additional_callbacks = []
    if not ray_tune:
        # PyTorch Lightning will handle it automatically
        pass
    else:
        additional_callbacks.append(
            TuneReportCheckpointCallback(
                # metrics={"val_loss": "val_loss", "val_accuracy": "val_accuracy"},
                metrics=["val_loss", "val_accuracy"],
                # filename="checkpoint", # (default)
            )
        )

    # Train
    print("---------------------------------------")
    print("Training ...")
    trainer = train_model(
        model,
        train_dl,
        val_dl,
        epochs=tunable_params["epochs"],
        additional_callbacks=additional_callbacks,
        enable_checkpointing=not ray_tune,
        enable_logging=not ray_tune,
        # TuneReportCheckpointCallback will handle checkpointing and logging
        use_gpu=fixed_params["use_gpu"],
        devices=fixed_params.get("devices", 1),
        verbose=fixed_params.get("verbose", False),
    )

    if trainer.is_global_zero and not ray_tune:  # Make sure we're at the root rank
        # Load every information we need from the trainer
        # (best_model, logger, log_dir)
        # nn_model = get_nn_model(tunable_params, fixed_params)
        # model = model.load_from_checkpoint(
        #     trainer.checkpoint_callback.best_model_path, nn_model=nn_model  # type: ignore
        # )
        model = LightningModuleWrapper.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path,  # type: ignore
            nn_model=get_nn_model(tunable_params, fixed_params),
        )
        logger = trainer.logger  # type: ignore
        log_dir = Path(logger.log_dir)  # type: ignore

        # Get the test version of trainer for testing and predicting (same logger)
        if not fixed_params["use_gpu"]:
            tester = L.Trainer(logger=logger, accelerator="cpu")  # type: ignore
        else:
            # Only use 1 GPU for testing and predicting
            tester = L.Trainer(logger=logger, accelerator="gpu", devices=1)  # type: ignore

        # Test
        print("---------------------------------------")
        print("Testing ...")
        print("### Train loss: ###")
        tester.test(model, dataloaders=train_dl)
        print("### Validation loss: ###")
        tester.test(model, dataloaders=val_dl)
        print("### Test loss: ###")  # The last one will overwrite the previous ones
        tester.test(model, dataloaders=test_dl)

        # Predict
        print("---------------------------------------")
        print("Predicting ...")
        plot_predictions(model, tester, test_dl, log_dir)


if __name__ == "__main__":
    """-----------------------------------------------"""
    model_name = "MLP"
    # model_name = "CNN"

    # use_gpu = False
    use_gpu = True

    devices = 1
    # devices = "auto" # use all available GPUs

    epochs = 3
    # epochs = 1

    # verbose = False
    verbose = True
    """-----------------------------------------------"""

    fixed_params = {
        "model_name": model_name,
        "loss": "cross_entropy",
        "metrics": ["cross_entropy", "accuracy"],
        # We must initialize the torchmetrics inside the model
        "use_gpu": use_gpu,  # if True, please use script to run the code
        # ------------------------------------------------------------
        # The above parameters are shared for both training and tuning
        # ------------------------------------------------------------
        "devices": devices,  # only used when use_gpu=True
        "verbose": verbose,
    }
    tunable_params = {
        "batch_size": 256,
        "optimizer": "Adam",
        "lr": 0.001,
        "l2_weight": 0.01,
        "epochs": epochs,
    }
    if fixed_params["model_name"] == "MLP":
        tunable_params["num_layers"] = 3
    elif fixed_params["model_name"] == "CNN":
        tunable_params["num_layers"] = 3

    # Set all random seeds (Python, NumPy, PyTorch)
    L.seed_everything(seed=42, workers=True)

    # Set the precision of the matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Train the model
    trainable(tunable_params, fixed_params, ray_tune=False)

    print("### Done ###")
