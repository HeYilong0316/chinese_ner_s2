import os
from glob import glob
import json

import numpy as np
import pandas as pd


def get_analyze(root_dir, 
                model, 
                file_name="log_history.json", 
                model_name="pytorch_model.bin", 
                k_fold=10, 
                select=False, 
                rm=False
                ):
    analyze_df = []
    for i in range(k_fold):
        k_fold_dir = f"{i}_fold"

        dir_path = os.path.join(root_dir, k_fold_dir)
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, "r", encoding="utf8") as r:
            log_history = json.load(r)
        
        max_score = 0
        max_epoch = None

        for log in log_history:
            score = log.get("eval_all_f_score", -1)

            if score > max_score:
                max_score = score
                max_epoch = log["epoch"]
                max_step = log["step"]

        if select:
            # 把最优模型copy出来
            model_path = os.path.join(dir_path, f"checkpoint-{max_step}", model_name)
            cmd = f"cp {model_path} {dir_path}"
            print("cmd: ", cmd)
            os.system(cmd)
        if rm:
            # 多余的模型删掉
            rm_path = os.path.join(dir_path, f"checkpoint-*")
            cmd = f"rm -r  {rm_path}"
            print("cmd: ", cmd)
            os.system(cmd)

        analyze_df.append({
            "model": model,
            "fold": i, 
            "max_score": max_score,
            "max_epoch": max_epoch

        })
    analyze_df = [{
        "model": model,
        "fold": "mean", 
        "max_score": np.mean([r["max_score"] for r in analyze_df]),
        "max_epoch": "-"

    }] + analyze_df
    analyze_df = pd.DataFrame(analyze_df)
    analyze_df = analyze_df[["model", "fold", "max_score", "max_epoch"]]
    return analyze_df.set_index("fold").T


if __name__ == "__main__":
    root_dir = "../user_data/output/output_layer_lstm_crf"
    model = "bert+words+layer+lstm+crf"
    file_name = "log_history.json"
    k_fold = 10
    analyze_df = get_analyze(root_dir, model, file_name, k_fold=k_fold)
    print(analyze_df.to_markdown())