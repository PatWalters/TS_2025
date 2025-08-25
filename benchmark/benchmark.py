#!/usr/bin/env python

import sys
import os
import pandas as pd
import json
import numpy as np
from glob import glob

SRC_PATH = "../src"
sys.path.append(SRC_PATH)

run_type = sys.argv[1]
if run_type == "TS":
    from ts_main import read_input, run_ts, parse_input_dict    
elif run_type == "RWS":
    from rws_run import run_rws
    from rws_main import parse_input_dict    
else:
    print("The first argument must be in [TS,RWS]")
    sys.exit(1)    

rxn_df = pd.read_csv("../reactions/reactions.csv")

num_runs = 10
for idx,[rxn_id, smarts] in enumerate(rxn_df.values):
    rxn_id = rxn_id.strip()
    num_components = len(smarts.split("."))

    ref_df = pd.read_parquet(f"../parquet/{num_components}_component/{rxn_id}.parquet")
    json_filename = f"../json/{num_components}_component/{rxn_id}_{run_type}.json"
    res_list = []
    with open(json_filename, 'r') as ifs:
        input_dict = json.load(ifs)    

    for query_idx,col in  enumerate(ref_df.columns[2:]):
        ref_df.sort_values(col,ascending=False,inplace=True)
        best_val = ref_df[col].values[0]
        ref_names = ref_df.head(100).Name.values
        found_list = []
        for j in range(0,num_runs):
            input_dict['evaluator_arg']['ref_colname'] = col
            parse_input_dict(input_dict)
            if run_type == "TS":
                res = run_ts(input_dict,hide_progress=True).sort_values("score",ascending=False).head(100)
            elif run_type == "RWS":
                res = run_rws(input_dict,hide_progress=True).sort_values("score",ascending=False).head(100)
            else:
                assert(False)
            found_list.append(len(res.query("Name in @ref_names")))
        print(rxn_id,col,best_val,found_list,np.mean(found_list),np.std(found_list))
        res_list.append([col,best_val]+found_list+[np.mean(found_list),np.std(found_list)])

    col_list = ["Query","Best"]+[f"run_{x:02d}" for x in range(0,num_runs)]+["Mean","Std"]
    res_df = pd.DataFrame(res_list,columns=col_list)
    res_df.to_csv(f"{rxn_id}_{run_type}_results.csv",index=False)
    

    




