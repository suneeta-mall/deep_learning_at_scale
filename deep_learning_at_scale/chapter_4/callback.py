from typing import Any, Dict, Literal

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from aim import Image as AimImage


class ImageLogger(pl.Callback):
    def __init__(self, plot_every_n_step: int = 3) -> None:
        self.plot_every_n_step = plot_every_n_step

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if batch_idx % self.plot_every_n_step == 0:
            self._log_image(
                trainer,
                pl_module.logger,
                batch,
                batch_idx,
                outputs["preds"],
                mode="train",
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.plot_every_n_step == 0:
            self._log_image(
                trainer,
                pl_module.logger,
                batch,
                batch_idx,
                outputs["preds"],
                mode="val",
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.plot_every_n_step == 0:
            self._log_image(
                trainer,
                pl_module.logger,
                batch,
                batch_idx,
                outputs["preds"],
                mode="test",
            )

    def _log_image(
        self,
        trainer: pl.Trainer,
        pl_logger: pl.loggers.Logger,
        batch: Dict[str, Any],
        batch_idx: int,
        pred: torch.Tensor,
        mode: Literal["train", "val", "test"],
    ):
        if batch_idx % self.plot_every_n_step == 0:
            step = batch_idx if mode == "test" else trainer.global_step
            pred = pred.detach().cpu()
            pl_logger.experiment.track(
                AimImage(
                    self._plot_sample_in_batch(batch, pred),
                    caption=f"""Epoch:{trainer.current_epoch} 
                    Batch:{batch_idx} Step:{step}""",
                ),
                name=f"{mode}/images",
                step=step,
                epoch=trainer.current_epoch,
                context={"subset": mode},
            )
            plt.close()

    def _plot_sample_in_batch(self, batch: Any, pred: torch.Tensor, idx: int = 0):
        fig, axs = plt.subplots(1, 4, figsize=(15, 15))

        pixel_values = batch[0].detach().cpu().numpy()
        mask_labels = batch[1].detach().cpu().numpy()
        pred_masks = pred[idx].numpy()

        image = pixel_values[idx, ...]
        image = image.transpose(1, 2, 0)
        axs[0].imshow(image)
        axs[0].title.set_text("Image")
        axs[1].imshow(mask_labels[idx, ...])
        axs[2].imshow(pred_masks)
        axs[3].imshow(image)
        axs[3].imshow(pred_masks, alpha=0.5)
        fig.tight_layout(pad=0.5)
        return fig
