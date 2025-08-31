"""
This module defines the Pydantic models for configuration management.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, FilePath, PositiveInt, PositiveFloat


class EvaluatorArgs(BaseModel):
    ref_filename: FilePath
    ref_colname: str


class BaseConfig(BaseModel):
    reagent_file_list: List[FilePath]
    reaction_smarts: str
    evaluator_class_name: str
    evaluator_arg: EvaluatorArgs
    log_filename: Optional[str] = None
    results_filename: Optional[str] = None
    hide_progress: bool = False


class TSConfig(BaseConfig):
    num_warmup_trials: PositiveInt = 5
    num_ts_iterations: PositiveInt = 10000
    ts_mode: str = "maximize"


class RWSConfig(BaseConfig):
    nprocesses: PositiveInt = 1
    min_cpds_per_core: PositiveInt = 50
    num_warmup_trials: PositiveInt = 5
    percent_of_library: PositiveFloat = 0.02
    scaling: int = 1
    stop: PositiveInt = 6000
