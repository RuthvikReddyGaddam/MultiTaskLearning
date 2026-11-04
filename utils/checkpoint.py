import os
import torch
import pandas as pd


def load_checkpoint(
    model,
    optimizer=None,
    path=None,
    device="cpu",
    scheduler=None,
):
    if path is None and isinstance(optimizer, (str, os.PathLike)):
        path = optimizer
        optimizer = None

    if device == "cpu" and hasattr(model, "parameters"):
        try:
            device = next(model.parameters()).device
        except Exception:
            pass

    checkpoint_data = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint_data["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict" in checkpoint_data
        and checkpoint_data["optimizer_state_dict"] is not None
    ):
        optimizer.load_state_dict(
            checkpoint_data["optimizer_state_dict"]
        )

    if (
        scheduler is not None
        and checkpoint_data.get("scheduler_state_dict") is not None
    ):
        scheduler.load_state_dict(
            checkpoint_data["scheduler_state_dict"]
        )

    epoch = checkpoint_data.get("epoch", 0)
    start_epoch = epoch + 1

    best_score = checkpoint_data.get(
        "best_score",
        None,
    )

    best_epoch = checkpoint_data.get(
        "best_epoch",
        None,
    )

    extra = checkpoint_data.get(
        "extra",
        None,
    )

    print(
        f"Checkpoint loaded from: {path}\n"
        f"Resuming from epoch: {start_epoch}"
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch,
        "start_epoch": start_epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "extra": extra,
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    filename,
    scheduler=None,
    best_score=None,
    best_epoch=None,
    extra=None,
):
    save_dict = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict() if optimizer is not None else None,
    }

    if scheduler is not None:
        save_dict[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    if best_score is not None:
        save_dict[
            "best_score"
        ] = best_score

    if best_epoch is not None:
        save_dict[
            "best_epoch"
        ] = best_epoch

    if extra is not None:
        save_dict[
            "extra"
        ] = extra

    checkpoint_directory = (
        os.path.dirname(filename)
    )

    if checkpoint_directory:
        os.makedirs(
            checkpoint_directory,
            exist_ok=True,
        )

    torch.save(
        save_dict,
        filename,
    )

    print(
        f"Checkpoint saved at epoch {epoch}: "
        f"{filename}"
    )


def save_train_history(
    history_path,
    epochs,
    train_losses,
    train_dice_scores,
    train_iou_scores,
    epoch_run_times,
    **kwargs,
):

    history_directory = (
        os.path.dirname(
            history_path
        )
    )

    if history_directory:

        os.makedirs(
            history_directory,
            exist_ok=True,
        )

    data = {
        "epoch":
            epochs,

        "train_loss":
            train_losses,

        "train_dice":
            train_dice_scores,

        "train_iou":
            train_iou_scores,

        "epoch_run_time":
            epoch_run_times,
    }
    data.update({k: v for k, v in kwargs.items() if v is not None})

    history_df = pd.DataFrame(data)

    history_df.to_csv(
        history_path,
        index=False,
    )