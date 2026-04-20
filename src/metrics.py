# src/metrics.py
from dataclasses import dataclass, field
import enum
from typing import List

from utils import avg


class MetricActions(enum.Enum):
    AVERAGE = enum.auto()
    TOTAL = enum.auto()
    PERCENT_AVERAGE = enum.auto()


@dataclass
class Metric:
    name: str
    actions: List[MetricActions]
    suffix: str=""
    decimal_precision: int=3
    data: list=field(default_factory=list)

    def __str__(self):
        res = ""
        for action in self.actions:
            match action:
                case MetricActions.AVERAGE:
                    res += f"avg. {self.name}: \
{avg(self.data):.{self.decimal_precision}f} {self.suffix}\n"
                case MetricActions.TOTAL:
                    res += f"tot. {self.name}: \
{sum(self.data):.{self.decimal_precision}f} {self.suffix}\n"
                case MetricActions.PERCENT_AVERAGE:
                    res += f"avg. {self.name}: \
{avg(self.data)*100:.{self.decimal_precision}f}%\n"
        return res
