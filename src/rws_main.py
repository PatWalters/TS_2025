#!/usr/bin/env python

import importlib
import sys
from datetime import timedelta
from timeit import default_timer as timer
from rws_utils import read_input
from rws_run import run_rws
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_input_dict(input_data: dict) -> None:
    """
    Parse the input dictionary and add the necessary information
    :param input_data:
    """
    module = importlib.import_module("evaluators")
    evaluator_class_name = input_data["evaluator_class_name"]
    class_ = getattr(module, evaluator_class_name)
    evaluator_arg = input_data["evaluator_arg"]
    evaluator = class_(evaluator_arg)
    input_data['evaluator_class'] = evaluator



def main():
    start = timer()
    json_filename = sys.argv[1]
    input_dict = read_input(json_filename)
    result_df = run_rws(input_dict)
    outfile_name = input_dict["results_filename"]
    result_df.sort_values("score",ascending=False).to_csv(outfile_name,index=False)
    end = timer()
    print("Elapsed time", timedelta(seconds=end - start))


if __name__ == "__main__":
    main()
