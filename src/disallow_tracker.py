"""
This module contains the DisallowTracker class, which is responsible for ensuring that
reagent combinations are not sampled more than once from a combinatorial library.
"""

import itertools
import random
from collections import defaultdict
from typing import DefaultDict, Set, Tuple

import numpy as np


class DisallowTracker:
    """
    A class to track and prevent the resampling of reagent combinations.

    This class is essential for sampling without replacement from a large virtual
    combinatorial library. It maintains a "disallow mask" to keep track of
    which combinations have already been evaluated.

    The internal state is managed using two sentinel values:
    - `Empty` (-1): Represents a component slot that has not yet been filled in a given combination.
    - `To_Fill` (None): Represents the specific component slot that is currently being selected.
    """

    Empty = -1
    To_Fill = None

    def __init__(self, reagent_counts: list[int]):
        """
        Initializes the DisallowTracker.

        :param reagent_counts: A list of the number of reagents for each component in the reaction.
                               For example, [10, 20, 34] for a 3-component reaction.
        """
        self._initial_reagent_counts = np.array(reagent_counts)
        self._reagent_exhaust_counts = self._get_reagent_exhaust_counts()

        # The core data structure for tracking disallowed combinations.
        # It maps a partial combination (a tuple of reagent indices) to a set of disallowed
        # indices for the component marked as `To_Fill`.
        self._disallow_mask: DefaultDict[Tuple[int | None], Set] = defaultdict(set)
        self._n_sampled = 0
        self._total_product_size = np.prod(reagent_counts)

    @property
    def n_cycles(self) -> int:
        """Returns the number of components in the reaction."""
        return len(self._initial_reagent_counts)

    def get_disallowed_selection_mask(self, current_selection: list[int | None]) -> Set[int]:
        """
        Returns the set of disallowed reagent indices for the component being filled.

        :param current_selection: A list representing the current partial selection. It must contain
                                  exactly one `To_Fill` value, indicating which component is being selected.
                                  Example: [12, None, 5] for a 3-component reaction where the second
                                  component is being selected.
        :return: A set of integer indices for the reagents that are disallowed for the current selection.
        """
        if len(current_selection) != self.n_cycles:
            raise ValueError(
                f"current_selection must be equal in length to number of sites ({self.n_cycles} for reaction)"
            )
        if len([v for v in current_selection if v == DisallowTracker.To_Fill]) != 1:
            raise ValueError("current_selection must have exactly one To_Fill slot.")

        return self._disallow_mask[tuple(current_selection)]

    def retire_one_synthon(self, cycle_id: int, synthon_index: int):
        """
        Retires a single synthon (reagent) from being selected in the future.
        This is typically used when a reagent is found to be invalid or unproductive.
        """
        retire_mask = [self.Empty] * self.n_cycles
        retire_mask[cycle_id] = synthon_index
        self._retire_synthon_mask(retire_mask=retire_mask)

    def _retire_synthon_mask(self, retire_mask: list[int]):
        """
        Recursively updates the disallow mask to retire a synthon or synthon combination.
        """
        # Base case: If the mask is fully specified (no Empty slots), it represents a single
        # product. We update the disallow mask for this product.
        if retire_mask.count(self.Empty) == 0:
            self._n_sampled += 1
            self._update(retire_mask)
        else:
            # Recursive step: For each unfilled slot, iterate through all possible reagents
            # and recursively call this function to retire all combinations involving the
            # original synthon to be retired.
            for cycle_id in [i for i in range(self.n_cycles) if retire_mask[i] == self.Empty]:
                retire_mask[cycle_id] = self.To_Fill
                ts_locations = np.ones(shape=self._initial_reagent_counts[cycle_id])
                disallowed_selections = self.get_disallowed_selection_mask(retire_mask)

                if len(disallowed_selections):
                    ts_locations[np.array(list(disallowed_selections))] = np.nan

                # For all valid remaining synthons, recursively retire them in combination
                # with the original synthon.
                for synthon_idx in np.argwhere(~np.isnan(ts_locations)).flatten():
                    retire_mask[cycle_id] = synthon_idx
                    self._retire_synthon_mask(retire_mask=retire_mask)

    def update(self, selected: list[int | None]) -> None:
        """
        Updates the disallow tracker with a newly sampled reagent combination.

        This ensures that this specific combination will not be sampled again.

        :param selected: A list of reagent indices representing the full product.
                         Example: [4, 5, 3] for a 3-component reaction.
        """
        if len(selected) != self.n_cycles:
            msg = f"DisallowTracker selected size {len(selected)} but reaction has {self.n_cycles} sites of diversity"
            raise ValueError(msg)
        for site_id, sel, max_size in zip(list(range(self.n_cycles)), selected, self._initial_reagent_counts):
            if sel is not None and sel >= max_size:
                raise ValueError(f"Disallowed given index {sel} for site {site_id} which has {max_size} reagents")

        self._update(selected)

    def sample(self) -> list[int]:
        """Randomly samples one valid product from the reaction without replacement."""
        if self._n_sampled == self._total_product_size:
            raise ValueError(
                f"Sampled {self._n_sampled} of {self._total_product_size} products in reaction - no more left to sample"
            )
        selection_mask: list[int | None] = [self.Empty] * self.n_cycles
        selection_order: list[int] = list(range(self.n_cycles))
        random.shuffle(selection_order)
        for cycle_id in selection_order:
            selection_mask[cycle_id] = DisallowTracker.To_Fill
            selection_candidate_scores = np.random.uniform(size=self._initial_reagent_counts[cycle_id])
            selection_candidate_scores[list(self._disallow_mask[tuple(selection_mask)])] = np.NaN
            selection_mask[cycle_id] = np.nanargmax(selection_candidate_scores).item(0)
        self.update(selection_mask)
        self._n_sampled += 1
        return selection_mask

    def _get_reagent_exhaust_counts(self) -> dict[tuple[int,], int]:
        """
        Calculates the number of combinations a reagent or reagent pair must be in
        before it is considered "exhausted". This is used to trigger the retirement
        of reagent combinations.

        Example for reagent_counts = [3, 4, 5]:
            - A reagent at site 0 is in 4 * 5 = 20 products.
            - A pair of reagents at sites (0, 2) is in 4 products.
        The returned dictionary stores these counts.
        """
        s = range(self.n_cycles)
        all_set = {*list(range(self.n_cycles))}
        power_set = itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1, self.n_cycles))
        return {p: np.prod(self._initial_reagent_counts[list(all_set - {*list(p)})]) for p in power_set}

    def _update(self, selected: list[int | None]):
        """
        Internal method to update the disallow masks without parameter checking.

        For a given selection (e.g., [4, 5, 3]), this method adds entries to the
        disallow mask for all partial combinations. For example, it will add:
        - 4 to the disallowed set for the key (None, 5, 3)
        - 5 to the disallowed set for the key (4, None, 3)
        - 3 to the disallowed set for the key (4, 5, None)
        """
        for idx, value in enumerate(selected):
            original_value = selected[idx]
            selected[idx] = self.To_Fill
            if value is not None and value not in self._disallow_mask[tuple(selected)]:
                if value != self.Empty:
                    self._disallow_mask[tuple(selected)].add(value)

                    # Check if a reagent or reagent combination is now exhausted.
                    count_key = tuple([r_pos for r_pos, r_idx in enumerate(selected) if r_idx != self.To_Fill])
                    if self._reagent_exhaust_counts[count_key] == len(self._disallow_mask[tuple(selected)]):
                        # If a combination is exhausted (e.g., all products containing the pair (reagent 5, reagent 3)
                        # have been sampled), we must recursively update the disallow mask for all sub-combinations.
                        # This is a complex but crucial step for correctness.
                        self._update([self.Empty if v == self.To_Fill else v for v in selected])
            selected[idx] = original_value
