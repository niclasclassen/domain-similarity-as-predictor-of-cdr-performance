import numpy as np
from recbole.evaluator import AbstractMetric, DataStruct
from recbole.utils import EvaluatorType


class CustomMetric(AbstractMetric):
    """
    TODO: Please follow https://recbole.io/docs/developer_guide/customize_metrics.html
    to create customized metrics.
    """
    metric_need = ['some_attribute']  # Temporary fix to avoid error
    metric_type = ['some_attribute']  # Temporary fix to avoid error

    def __init__(self, config):
        super().__init__(config)

    def calculate_metric(self, dataobject):
        pass
