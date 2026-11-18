import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.features import (
    bureau_features,
    card_features,
    installment_features,
    merge_feature_tables,
    pos_features,
    prepare_application,
    previous_features,
)


def build_dataset(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = []

    def read(name):
        table = pd.read_csv(data_dir / f"{name}.csv")
        quality.append(
            {
                "table": name,
                "rows": len(table),
                "columns": len(table.columns),
                "missing_cells": int(table.isna().sum().sum()),
                "clients": table["SK_ID_CURR"].nunique()
                if "SK_ID_CURR" in table
                else None,
            }
        )
        return table

    feature_tables = {}
    feature_tables["BUREAU"] = bureau_features(read("bureau"), read("bureau_balance"))
    gc.collect()
    for prefix, name, aggregate in [
        ("POS", "POS_CASH_balance", pos_features),
        ("INST", "installments_payments", installment_features),
        ("CC", "credit_card_balance", card_features),
        ("PREV", "previous_application", previous_features),
    ]:
        feature_tables[prefix] = aggregate(read(name))
        print(prefix, feature_tables[prefix].shape, flush=True)
        gc.collect()
    application = read("application_train")
    result = merge_feature_tables(prepare_application(application), feature_tables)
    result.to_parquet(output_dir / "train_features.parquet", index=False)
    pd.DataFrame(quality).to_csv(output_dir / "source_quality.csv", index=False)
    result.isna().mean().sort_values(ascending=False).rename("missing_share").to_csv(
        output_dir / "feature_missingness.csv",
    )
    audit = {
        "rows_before_join": len(application),
        "rows_after_join": len(result),
        "duplicate_client_ids": int(result["SK_ID_CURR"].duplicated().sum()),
        "missing_target": int(result["TARGET"].isna().sum()),
        "default_rate": float(result["TARGET"].mean()),
        "infinite_values": int(np.isinf(result.select_dtypes("number")).sum().sum()),
        "employment_placeholder_count": int(
            application["DAYS_EMPLOYED"].eq(365243).sum()
        ),
        "zero_income": int(application["AMT_INCOME_TOTAL"].eq(0).sum()),
    }
    (output_dir / "data_quality.json").write_text(json.dumps(audit, indent=2))
    print(audit, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed"))
    arguments = parser.parse_args()
    build_dataset(arguments.data_dir, arguments.output_dir)
