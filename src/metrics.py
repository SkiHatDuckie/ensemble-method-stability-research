# src/metrics.py
from dataclasses import dataclass
import enum


class MetricActions(enum.Enum):
    AVERAGE = enum.auto()
    TOTAL = enum.auto()


@dataclass
class Metric:
    name: str
    data: list=[]
    actions: MetricActions|list[MetricActions]
