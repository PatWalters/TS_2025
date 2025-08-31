import functools
import math
import random
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from disallow_tracker import DisallowTracker
from evaluators import DBEvaluator, LookupEvaluator
from reagent import Reagent
from ts_logger import get_logger
from ts_utils import read_reagents


class ThompsonSampler:
    def __init__(self, mode="maximize", log_filename: Optional[str] = None):
        """
        Basic init
        :param mode: maximize or minimize
        :param log_filename: Optional filename to write logging to. If None, logging will be output to stdout
        """
        # A list of lists of Reagents. Each component in the reaction will have one list of Reagents in this list
        self.reagent_lists: List[List[Reagent]] = []
        self.reaction = None
        self.evaluator = None
        self.num_prods = 0
        self.logger = get_logger(__name__, filename=log_filename)
        self._disallow_tracker = None
        self.hide_progress = False
        self._mode = mode
        if self._mode == "maximize":
            self.pick_function = np.nanargmax
            self._top_func = max
        elif self._mode == "minimize":
            self.pick_function = np.nanargmin
            self._top_func = min
        elif self._mode == "maximize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = max
        elif self._mode == "minimize_boltzmann":
            # See documentation for _boltzmann_reweighted_pick
            self.pick_function = functools.partial(self._boltzmann_reweighted_pick)
            self._top_func = min
        else:
            raise ValueError(f"{mode} is not a supported argument")
        self._warmup_std = None

    def _boltzmann_reweighted_pick(self, scores: np.ndarray):
        """Rather than choosing the top sampled score, use a reweighted probability.

        Zhao, H., Nittinger, E. & Tyrchan, C. Enhanced Thompson Sampling by Roulette
        Wheel Selection for Screening Ultra-Large Combinatorial Libraries.
        bioRxiv 2024.05.16.594622 (2024) doi:10.1101/2024.05.16.594622
        suggested several modifications to the Thompson Sampling procedure.
        This method implements one of those, namely a Boltzmann style probability distribution
        from the sampled values. The reagent is chosen based on that distribution rather than
        simply the max sample.
        """
        if self._mode == "minimize_boltzmann":
            scores = -scores
        exp_terms = np.exp(scores / self._warmup_std)
        probs = exp_terms / np.nansum(exp_terms)
        probs[np.isnan(probs)] = 0.0
        return np.random.choice(probs.shape[0], p=probs)

    def set_hide_progress(self, hide_progress: bool) -> None:
        """
        Hide the progress bars
        :param hide_progress: set to True to hide the progress baars
        """
        self.hide_progress = hide_progress

    def read_reagents(self, reagent_file_list, num_to_select: Optional[int] = None):
        """
        Reads the reagents from reagent_file_list
        :param reagent_file_list: List of reagent filepaths
        :param num_to_select: Max number of reagents to select from the reagents file (for dev purposes only)
        :return: None
        """
        self.reagent_lists = read_reagents(reagent_file_list, num_to_select)
        self.num_prods = math.prod([len(x) for x in self.reagent_lists])
        self.logger.info(f"{self.num_prods:.2e} possible products")
        self._disallow_tracker = DisallowTracker([len(x) for x in self.reagent_lists])

    def get_num_prods(self) -> int:
        """
        Get the total number of possible products
        :return: num_prods
        """
        return self.num_prods

    def set_evaluator(self, evaluator):
        """
        Define the evaluator
        :param evaluator: evaluator class, must define an evaluate method that takes an RDKit molecule
        """
        self.evaluator = evaluator

    def set_reaction(self, rxn_smarts):
        """
        Define the reaction
        :param rxn_smarts: reaction SMARTS
        """
        self.reaction = AllChem.ReactionFromSmarts(rxn_smarts)

    def evaluate(self, choice_list: List[int]) -> Tuple[str, str, float]:
        """Evaluate a set of reagents
        :param choice_list: list of reagent ids
        :return: smiles for the reaction product, score for the reaction product
        """
        selected_reagents = []
        for idx, choice in enumerate(choice_list):
            component_reagent_list = self.reagent_lists[idx]
            selected_reagents.append(component_reagent_list[choice])
        prod = self.reaction.RunReactants([reagent.mol for reagent in selected_reagents])
        product_name = "_".join([reagent.reagent_name for reagent in selected_reagents])
        res = np.nan
        product_smiles = "FAIL"
        if prod:
            prod_mol = prod[0][0]  # RunReactants returns Tuple[Tuple[Mol]]
            Chem.SanitizeMol(prod_mol)
            product_smiles = Chem.MolToSmiles(prod_mol)
            if isinstance(self.evaluator, DBEvaluator) or isinstance(self.evaluator, LookupEvaluator):
                res = self.evaluator.evaluate(product_name)
                res = float(res)
            else:
                res = self.evaluator.evaluate(prod_mol)
            if np.isfinite(res):
                [reagent.add_score(res) for reagent in selected_reagents]
        return product_smiles, product_name, res

    def _perform_warmup_trial(self, component_idx: int, reagent_idx: int) -> Optional[List]:
        """Perform a single warmup trial for a specific reagent."""
        reagent_count_list = [len(x) for x in self.reagent_lists]
        partner_list = [x for x in range(len(self.reagent_lists)) if x != component_idx]

        current_list = [DisallowTracker.Empty] * len(self.reagent_lists)
        current_list[component_idx] = DisallowTracker.To_Fill
        disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)

        if reagent_idx not in disallow_mask:
            current_list[component_idx] = reagent_idx
            # Randomly select reagents for each additional component
            for p_idx in partner_list:
                current_list[p_idx] = DisallowTracker.To_Fill
                disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(current_list)
                selection_scores = np.random.uniform(size=reagent_count_list[p_idx])
                selection_scores[list(disallow_mask)] = np.nan
                current_list[p_idx] = np.nanargmax(selection_scores).item(0)

            self._disallow_tracker.update(current_list)
            product_smiles, product_name, score = self.evaluate(current_list)
            if np.isfinite(score):
                return [score, product_smiles, product_name]
        return None

    def _initialize_reagents(self, warmup_scores: List[float]):
        """Initialize reagent scores based on warmup results."""
        if not warmup_scores:
            self.logger.warning("No successful evaluations during warmup. Reagents will not be initialized.")
            return

        prior_mean = np.mean(warmup_scores)
        prior_std = np.std(warmup_scores)
        self._warmup_std = prior_std

        for i, reagent_list in enumerate(self.reagent_lists):
            for j, reagent in enumerate(reagent_list):
                try:
                    reagent.init_given_prior(prior_mean=prior_mean, prior_std=prior_std)
                except ValueError:
                    self.logger.info(
                        f"Skipping reagent {reagent.reagent_name} because there were no "
                        f"successful evaluations during warmup"
                    )
                    self._disallow_tracker.retire_one_synthon(i, j)

    def warm_up(self, num_warmup_trials=3):
        """Warm-up phase, each reagent is sampled with num_warmup_trials random partners
        :param num_warmup_trials: number of times to sample each reagent
        """
        warmup_results = []
        num_components = len(self.reagent_lists)

        for i in range(num_components):
            num_reagents = len(self.reagent_lists[i])
            for j in tqdm(range(num_reagents), desc=f"Warmup {i + 1} of {num_components}", disable=self.hide_progress):
                for _ in range(num_warmup_trials):
                    result = self._perform_warmup_trial(component_idx=i, reagent_idx=j)
                    if result:
                        warmup_results.append(result)

        warmup_scores = [res[0] for res in warmup_results]
        self.logger.info(
            f"warmup score stats: "
            f"cnt={len(warmup_scores)}, "
            f"mean={np.mean(warmup_scores) if warmup_scores else 0:.4f}, "
            f"std={np.std(warmup_scores) if warmup_scores else 0:.4f}, "
            f"min={np.min(warmup_scores) if warmup_scores else 0:.4f}, "
            f"max={np.max(warmup_scores) if warmup_scores else 0:.4f}"
        )

        self._initialize_reagents(warmup_scores)

        if warmup_scores:
            self.logger.info(f"Top score found during warmup: {max(warmup_scores):.3f}")
        return warmup_results

    def search(self, num_cycles=25):
        """Run the search
        :param: num_cycles: number of search iterations
        :return: a list of SMILES and scores
        """
        # Note on performance: The main computational cost of this loop is the `evaluate` method,
        # which typically involves expensive chemical simulations or model predictions.
        # The overhead of the loop itself (e.g., creating numpy arrays) is negligible in comparison.
        # Any significant performance optimization efforts should focus on the evaluator.
        out_list = []
        rng = np.random.default_rng()
        for i in tqdm(range(0, num_cycles), desc="Cycle", disable=self.hide_progress):
            selected_reagents = [DisallowTracker.Empty] * len(self.reagent_lists)
            for cycle_id in random.sample(range(0, len(self.reagent_lists)), len(self.reagent_lists)):
                reagent_list = self.reagent_lists[cycle_id]
                selected_reagents[cycle_id] = DisallowTracker.To_Fill
                disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
                stds = np.array([r.current_std for r in reagent_list])
                mu = np.array([r.current_mean for r in reagent_list])
                choice_row = rng.normal(size=len(reagent_list)) * stds + mu
                if disallow_mask:
                    choice_row[np.array(list(disallow_mask))] = np.nan
                selected_reagents[cycle_id] = self.pick_function(choice_row)
            self._disallow_tracker.update(selected_reagents)
            # Select a reagent for each component, according to the choice function
            smiles, name, score = self.evaluate(selected_reagents)
            if np.isfinite(score):
                out_list.append([score, smiles, name])
            if i % 100 == 0 and len(out_list) > 0:
                top_score, top_smiles, top_name = self._top_func(out_list)
                self.logger.info(f"Iteration: {i} max score: {top_score:2f} smiles: {top_smiles} {top_name}")
        return out_list
