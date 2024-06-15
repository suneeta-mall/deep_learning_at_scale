from collections import defaultdict

from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.logger import rank_zero_experiment
from lightning.pytorch.utilities import rank_zero_only


class DictLogger(Logger):
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(lambda: [])

    @property
    def name(self):
        return "DictLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    def version(self):
        return "1.0"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for key in metrics.keys():
            self.metrics[key].append(metrics[key])

    @rank_zero_only
    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status):
        pass
