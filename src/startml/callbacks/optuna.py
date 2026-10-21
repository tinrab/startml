import optuna
from lightning import Callback, LightningModule, Trainer


class OptunaPruning(Callback):
    """Report a validation metric to an Optuna trial after each epoch and stop if it's pruned.

    Training is stopped gracefully (`trainer.should_stop`) rather than by raising, so loggers and
    checkpoints are finalized; the caller checks `pruned` afterwards and raises `TrialPruned`.
    """

    def __init__(self, trial: optuna.Trial, monitor: str) -> None:
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.pruned = False

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics.get(self.monitor)
        if score is None:
            return
        epoch = trainer.current_epoch
        self.trial.report(float(score), step=epoch)
        if self.trial.should_prune():
            self.pruned = True
            trainer.should_stop = True
