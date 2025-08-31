import pandas as pd
import importlib

from config import RWSConfig
from rws_sampling import RWSSampler
from ts_logger import get_logger


def run_rws(config: RWSConfig, hide_progress=True) -> pd.DataFrame:
    """
    Run a Thompson sampling with roulette wheel selection
    :param hide_progress: hide the progress bar
    :param config: RWSConfig object with input parameters
    """
    # Dynamically load the evaluator
    module = importlib.import_module("evaluators")
    evaluator_class = getattr(module, config.evaluator_class_name)
    evaluator = evaluator_class(config.evaluator_arg.dict())

    logger = get_logger(__name__, filename=config.log_filename)

    # setup ts
    ts = RWSSampler(config.nprocesses, config.scaling)
    ts.set_hide_progress(config.hide_progress)
    ts.set_evaluator(evaluator)
    ts.read_reagents([str(p) for p in config.reagent_file_list], num_to_select=None)
    ts.set_reaction(config.reaction_smarts)
    # run the warm-up phase to generate an initial set of scores for each reagent
    nw = ts.warm_up(config.num_warmup_trials, config.results_filename)
    # run the search with TS
    out_list = ts.search(
        percent_of_library=config.percent_of_library,
        min_cpds_per_core=config.min_cpds_per_core,
        stop=config.stop,
        results_filename=config.results_filename,
    )

    # logging
    total_evaluations = len(out_list) + nw
    percent_searched = total_evaluations / ts.get_num_prods() * 100
    logger.info(f"{total_evaluations} evaluations | {percent_searched:.3f}% of total")
    logger.info(f"Saved results to: {config.results_filename}")
    out_df = pd.DataFrame(out_list, columns=["score", "SMILES", "Name"])
    if not config.hide_progress:
        if config.scaling > 0:
            print(out_df.sort_values("score", ascending=False).drop_duplicates(subset="SMILES").head(100))
        else:
            print(out_df.sort_values("score", ascending=True).drop_duplicates(subset="SMILES").head(100))
    return out_df
