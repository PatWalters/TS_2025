#!/usr/bin/env python

import sys
import importlib
from datetime import timedelta
from timeit import default_timer as timer

from rdkit import RDLogger

from config import RWSConfig
from rws_run import run_rws
from ts_utils import load_config

RDLogger.DisableLog("rdApp.*")


def main():
    start = timer()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} infile.json")
        sys.exit(1)

    json_filename = sys.argv[1]
    config = load_config(json_filename)
    if not isinstance(config, RWSConfig):
        print(f"Error: {json_filename} is not a valid RWS config file.")
        sys.exit(1)

    # Dynamically load the evaluator
    module = importlib.import_module("evaluators")
    evaluator_class = getattr(module, config.evaluator_class_name)
    config.evaluator_class = evaluator_class(config.evaluator_arg.dict())

    result_df = run_rws(config)
    if config.results_filename:
        result_df.sort_values("score", ascending=False).to_csv(config.results_filename, index=False)

    end = timer()
    print("Elapsed time", timedelta(seconds=end - start))


if __name__ == "__main__":
    main()
