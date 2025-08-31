#!/usr/bin/env python

import sys
import importlib
from datetime import timedelta
from timeit import default_timer as timer

import pandas as pd

from config import TSConfig
from thompson_sampling import ThompsonSampler
from ts_logger import get_logger
from ts_utils import load_config


def run_ts(config: TSConfig, hide_progress: bool = False) -> pd.DataFrame:
    """
    Perform a Thompson sampling run based on the provided configuration.
    """
    # Dynamically load the evaluator
    module = importlib.import_module("evaluators")
    evaluator_class = getattr(module, config.evaluator_class_name)
    evaluator = evaluator_class(config.evaluator_arg.dict())

    logger = get_logger(__name__, filename=config.log_filename)

    ts = ThompsonSampler(mode=config.ts_mode, log_filename=config.log_filename)
    ts.set_hide_progress(hide_progress)
    ts.set_evaluator(evaluator)
    ts.read_reagents(reagent_file_list=[str(p) for p in config.reagent_file_list], num_to_select=None)
    ts.set_reaction(config.reaction_smarts)
    ts.warm_up(num_warmup_trials=config.num_warmup_trials)
    out_list = ts.search(num_cycles=config.num_ts_iterations)

    total_evaluations = evaluator.counter
    percent_searched = total_evaluations / ts.get_num_prods() * 100
    logger.info(f"{total_evaluations} evaluations | {percent_searched:.3f}% of total")

    out_df = pd.DataFrame(out_list, columns=["score", "SMILES", "Name"])
    if config.results_filename:
        out_df.to_csv(config.results_filename, index=False)
        logger.info(f"Saved results to: {config.results_filename}")

    if not hide_progress:
        sort_ascending = False if config.ts_mode == "maximize" else True
        print(out_df.sort_values("score", ascending=sort_ascending).drop_duplicates(subset="SMILES").head(10))

    return out_df


def main():
    start = timer()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} infile.json")
        sys.exit(1)

    json_filename = sys.argv[1]
    config = load_config(json_filename)
    if not isinstance(config, TSConfig):
        print(f"Error: {json_filename} is not a valid Thompson Sampling config file.")
        sys.exit(1)

    run_ts(config)
    end = timer()
    print("Elapsed time", timedelta(seconds=end - start))


if __name__ == "__main__":
    main()
