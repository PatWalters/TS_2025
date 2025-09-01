This repository contains the code to accompany the paper "Enhanced Thompson Sampling by Roulette Wheel Selection for Screening Ultralarge Combinatorial Libraries".  The paper, and the code in this repo, compares Thompson Sampling (TS) and Roulette Wheel Sampling (RWS) for shape-based virtual screening using ROCS.  This repo combines code from the [original TS codebase](https://github.com/PatWalters/TS) and [Hongtao Zhao's RW_TS code](https://github.com/WIMNZhao/TS_Enhanced).

## A Couple of Caveats
This repo is big, the total content is 5.6GB.  Most of that is the parquet files that hold the results from 2,180 ROCS searches.  To download the parquet files, you need to first install [GitHub Large File Storage (LFS)](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

## Directory Structure

```
.
├── analysis/         # Jupyter notebooks for analyzing results
├── benchmark/        # Scripts for performance benchmarking
├── json/             # Input JSON configuration files
├── parquet/          # Processed data in Parquet format
├── reactions/        # Raw reagent and product data (.smi files)
├── results/          # Output CSV files with high-scoring molecules
├── src/              # Python source code for the core logic
├── LICENSE
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:PatWalters/TS_2025.git
    cd TS_2025
    ```

2.  **Create a Python virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```

## How to Run TS or RWS

The screening process is controlled by a JSON configuration file. Templates and examples can be found in the `json/` directory.

1.  **Configure your run**: Copy an existing JSON file (e.g., `json/template_TS.json`) and edit it:
    -   `reagent_file_list`: Point to the `.smi` files for each component in your reaction.
    -   `reaction_smarts`: Provide the reaction SMARTS string.
    -   `evaluator_class_name`: Specify the scoring function to use from `src/evaluators.py`.
    -   `results_filename`: Define the output path for the results CSV.
    -   `log_filename`: Define the output path for the log file.

    Please note that the JSON file formats for TS and RWS are not the same. 

2.  **Execute the screening:**

    -   **For Thompson Sampling:**
        ```bash
        python src/ts_main.py json/your_config_file.json
        ```

    -   **For Roulette Wheel Selection:**
        ```bash
        python src/rws_main.py json/your_config_file.json
        ```

3.  **Analyze the results**: The output will be saved to the CSV file specified in your configuration.

## Data Pipeline

The typical workflow for data is as follows:

1.  **Raw Reagents (`reactions/`)**: Reagent data is stored in SMILES format (`.smi`).
2.  **Input Configuration (`json/`)**: JSON files define how to combine reagents and evaluate the products. The `prepare_input.ipynb` notebook can be used to help generate these files.
3.  **Execution**: The `ts_main.py` or `rws_main.py` scripts consume the JSON files and reagent data.
4.  **Output Results (`results/`)**: The top-scoring molecules are written to CSV files.
