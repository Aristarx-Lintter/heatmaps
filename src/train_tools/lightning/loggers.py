import pytorch_lightning as pl
from clearml import Task


class ClearMLLightningLogger(pl.loggers.Logger):
    def __init__(self, project_name: str, task_name: str):
        super().__init__()
        self.task = Task.init(project_name=project_name, task_name=task_name)

    @property
    def experiment(self):
        return self.task

    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.task.get_logger().report_scalar(title=key, series=key, value=value, iteration=step)

    def log_hyperparams(self, params):
        self.task.connect(params)

    @property
    def name(self):
        return self.task.name

    @property
    def version(self):
        return self.task.id
