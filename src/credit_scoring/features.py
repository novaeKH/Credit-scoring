"""Client-level aggregates retained from the original research notebook."""

import numpy as np
import pandas as pd

ID_COLUMN = "SK_ID_CURR"


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def prepare_application(application: pd.DataFrame) -> pd.DataFrame:
    result = application.copy()
    result["DAYS_EMPLOYED"] = result["DAYS_EMPLOYED"].replace(365243, np.nan)
    result["APP_CREDIT_INCOME_RATIO"] = safe_ratio(
        result["AMT_CREDIT"], result["AMT_INCOME_TOTAL"]
    )
    result["APP_ANNUITY_INCOME_RATIO"] = safe_ratio(
        result["AMT_ANNUITY"], result["AMT_INCOME_TOTAL"]
    )
    result["APP_CREDIT_ANNUITY_RATIO"] = safe_ratio(
        result["AMT_CREDIT"], result["AMT_ANNUITY"]
    )
    result["APP_GOODS_CREDIT_RATIO"] = safe_ratio(
        result["AMT_GOODS_PRICE"], result["AMT_CREDIT"]
    )
    result["APP_EMPLOYED_AGE_RATIO"] = safe_ratio(
        result["DAYS_EMPLOYED"].abs(), result["DAYS_BIRTH"].abs()
    )
    return result


def bureau_features(bureau, bureau_balance) -> pd.DataFrame:
    bureau_balance_history = bureau_balance.copy()
    bureau_balance_history["BUREAU_IS_OVERDUE"] = (
        bureau_balance_history["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
    )
    bureau_balance_history["BUREAU_IS_SEVERE_OVERDUE"] = (
        bureau_balance_history["STATUS"].isin(["3", "4", "5"]).astype("int8")
    )
    bureau_balance_by_credit = (
        bureau_balance_history.groupby("SK_ID_BUREAU")
        .agg(
            BUREAU_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            BUREAU_OVERDUE_MONTHS=("BUREAU_IS_OVERDUE", "sum"),
            BUREAU_SEVERE_OVERDUE_MONTHS=("BUREAU_IS_SEVERE_OVERDUE", "sum"),
        )
        .reset_index()
    )
    bureau_balance_by_credit["BUREAU_OVERDUE_RATE"] = (
        bureau_balance_by_credit["BUREAU_OVERDUE_MONTHS"]
        / bureau_balance_by_credit["BUREAU_MONTHS_COUNT"]
    )
    bureau_with_balance = bureau.merge(
        bureau_balance_by_credit, on="SK_ID_BUREAU", how="left", validate="one_to_one"
    )
    bureau_with_balance["BUREAU_IS_ACTIVE"] = (
        bureau_with_balance["CREDIT_ACTIVE"] == "Active"
    ).astype("int8")
    bureau_with_balance["BUREAU_IS_CLOSED"] = (
        bureau_with_balance["CREDIT_ACTIVE"] == "Closed"
    ).astype("int8")
    bureau_with_balance["BUREAU_DEBT_CREDIT_RATIO"] = safe_ratio(
        bureau_with_balance["AMT_CREDIT_SUM_DEBT"],
        bureau_with_balance["AMT_CREDIT_SUM"],
    )
    client_bureau_features = (
        bureau_with_balance.groupby("SK_ID_CURR")
        .agg(
            BUREAU_CREDIT_COUNT=("SK_ID_BUREAU", "count"),
            BUREAU_ACTIVE_CREDIT_COUNT=("BUREAU_IS_ACTIVE", "sum"),
            BUREAU_CLOSED_CREDIT_COUNT=("BUREAU_IS_CLOSED", "sum"),
            BUREAU_TOTAL_MONTHS=("BUREAU_MONTHS_COUNT", "sum"),
            BUREAU_TOTAL_OVERDUE_MONTHS=("BUREAU_OVERDUE_MONTHS", "sum"),
            BUREAU_TOTAL_SEVERE_OVERDUE_MONTHS=("BUREAU_SEVERE_OVERDUE_MONTHS", "sum"),
            BUREAU_MEAN_OVERDUE_RATE=("BUREAU_OVERDUE_RATE", "mean"),
            BUREAU_MAX_OVERDUE_RATE=("BUREAU_OVERDUE_RATE", "max"),
            BUREAU_MEAN_DEBT_CREDIT_RATIO=("BUREAU_DEBT_CREDIT_RATIO", "mean"),
            BUREAU_MAX_DEBT_CREDIT_RATIO=("BUREAU_DEBT_CREDIT_RATIO", "max"),
            BUREAU_TOTAL_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
            BUREAU_TOTAL_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
            BUREAU_MAX_CREDIT_OVERDUE=("AMT_CREDIT_MAX_OVERDUE", "max"),
            BUREAU_LAST_CREDIT_DAYS=("DAYS_CREDIT", "max"),
        )
        .reset_index()
    )
    client_bureau_features["BUREAU_ACTIVE_CREDIT_RATE"] = safe_ratio(
        client_bureau_features["BUREAU_ACTIVE_CREDIT_COUNT"],
        client_bureau_features["BUREAU_CREDIT_COUNT"],
    )
    return client_bureau_features


def pos_features(pos_cash_balance) -> pd.DataFrame:
    pos_history = pos_cash_balance.copy()
    pos_history["POS_IS_DPD"] = (pos_history["SK_DPD"] > 0).astype("int8")
    pos_history["POS_IS_DPD_DEF"] = (pos_history["SK_DPD_DEF"] > 0).astype("int8")
    pos_history["POS_IS_DPD_30_PLUS"] = (pos_history["SK_DPD"] >= 30).astype("int8")
    last_pos_records = (
        pos_history.sort_values(["SK_ID_PREV", "MONTHS_BALANCE"])
        .groupby("SK_ID_PREV", as_index=False)
        .tail(1)[
            [
                "SK_ID_PREV",
                "NAME_CONTRACT_STATUS",
                "CNT_INSTALMENT_FUTURE",
                "MONTHS_BALANCE",
            ]
        ]
        .rename(
            columns={
                "NAME_CONTRACT_STATUS": "POS_LAST_STATUS",
                "CNT_INSTALMENT_FUTURE": "POS_LAST_FUTURE_INSTALLMENTS",
                "MONTHS_BALANCE": "POS_LAST_MONTH",
            }
        )
    )
    pos_credit_features = (
        pos_history.groupby(["SK_ID_PREV", "SK_ID_CURR"])
        .agg(
            POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            POS_DPD_MONTHS=("POS_IS_DPD", "sum"),
            POS_DPD_DEF_MONTHS=("POS_IS_DPD_DEF", "sum"),
            POS_DPD_30_PLUS_MONTHS=("POS_IS_DPD_30_PLUS", "sum"),
            POS_MAX_DPD=("SK_DPD", "max"),
            POS_MEAN_DPD=("SK_DPD", "mean"),
            POS_MAX_DPD_DEF=("SK_DPD_DEF", "max"),
            POS_MAX_INSTALMENT_COUNT=("CNT_INSTALMENT", "max"),
        )
        .reset_index()
    )
    pos_credit_features["POS_DPD_RATE"] = (
        pos_credit_features["POS_DPD_MONTHS"] / pos_credit_features["POS_MONTHS_COUNT"]
    )
    pos_credit_features["POS_DPD_DEF_RATE"] = (
        pos_credit_features["POS_DPD_DEF_MONTHS"]
        / pos_credit_features["POS_MONTHS_COUNT"]
    )
    pos_credit_features["POS_DPD_30_PLUS_RATE"] = (
        pos_credit_features["POS_DPD_30_PLUS_MONTHS"]
        / pos_credit_features["POS_MONTHS_COUNT"]
    )
    pos_credit_features = pos_credit_features.merge(
        last_pos_records, on="SK_ID_PREV", how="left", validate="one_to_one"
    )
    pos_credit_features["POS_IS_COMPLETED"] = (
        pos_credit_features["POS_LAST_STATUS"] == "Completed"
    ).astype("int8")
    pos_credit_features["POS_IS_ACTIVE"] = (
        pos_credit_features["POS_LAST_STATUS"] == "Active"
    ).astype("int8")
    risky_pos_statuses = ["Demand", "Amortized debt", "Returned to the store"]
    pos_credit_features["POS_HAS_RISKY_STATUS"] = (
        pos_credit_features["POS_LAST_STATUS"].isin(risky_pos_statuses).astype("int8")
    )
    client_pos_features = (
        pos_credit_features.groupby("SK_ID_CURR")
        .agg(
            POS_CONTRACT_COUNT=("SK_ID_PREV", "count"),
            POS_TOTAL_MONTHS=("POS_MONTHS_COUNT", "sum"),
            POS_TOTAL_DPD_MONTHS=("POS_DPD_MONTHS", "sum"),
            POS_TOTAL_DPD_30_PLUS_MONTHS=("POS_DPD_30_PLUS_MONTHS", "sum"),
            POS_MEAN_DPD_RATE=("POS_DPD_RATE", "mean"),
            POS_MAX_DPD_RATE=("POS_DPD_RATE", "max"),
            POS_MEAN_DPD_DEF_RATE=("POS_DPD_DEF_RATE", "mean"),
            POS_MAX_DPD_30_PLUS_RATE=("POS_DPD_30_PLUS_RATE", "max"),
            POS_MAX_DPD=("POS_MAX_DPD", "max"),
            POS_MEAN_MAX_DPD=("POS_MAX_DPD", "mean"),
            POS_COMPLETED_CONTRACTS=("POS_IS_COMPLETED", "sum"),
            POS_ACTIVE_CONTRACTS=("POS_IS_ACTIVE", "sum"),
            POS_RISKY_STATUS_CONTRACTS=("POS_HAS_RISKY_STATUS", "sum"),
            POS_MEAN_FUTURE_INSTALLMENTS=("POS_LAST_FUTURE_INSTALLMENTS", "mean"),
            POS_MAX_INSTALMENT_COUNT=("POS_MAX_INSTALMENT_COUNT", "max"),
        )
        .reset_index()
    )
    client_pos_features["POS_COMPLETED_RATE"] = safe_ratio(
        client_pos_features["POS_COMPLETED_CONTRACTS"],
        client_pos_features["POS_CONTRACT_COUNT"],
    )
    client_pos_features["POS_ACTIVE_RATE"] = safe_ratio(
        client_pos_features["POS_ACTIVE_CONTRACTS"],
        client_pos_features["POS_CONTRACT_COUNT"],
    )
    client_pos_features["POS_RISKY_STATUS_RATE"] = safe_ratio(
        client_pos_features["POS_RISKY_STATUS_CONTRACTS"],
        client_pos_features["POS_CONTRACT_COUNT"],
    )
    return client_pos_features


def installment_features(installments_payments) -> pd.DataFrame:
    installment_level = installments_payments.groupby(
        ["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"],
        as_index=False,
    ).agg(
        DAYS_INSTALMENT=("DAYS_INSTALMENT", "max"),
        DAYS_LAST_PAYMENT=("DAYS_ENTRY_PAYMENT", "max"),
        AMT_INSTALMENT=("AMT_INSTALMENT", "max"),
        AMT_PAYMENT_TOTAL=("AMT_PAYMENT", "sum"),
        PAYMENT_PARTS_COUNT=("AMT_PAYMENT", "count"),
    )
    installment_level["INST_PAYMENT_DELAY_DAYS"] = (
        installment_level["DAYS_LAST_PAYMENT"] - installment_level["DAYS_INSTALMENT"]
    ).clip(lower=0)
    installment_level["INST_IS_LATE_PAYMENT"] = (
        installment_level["INST_PAYMENT_DELAY_DAYS"] > 0
    ).astype("int8")
    installment_level["INST_PAYMENT_SHORTFALL"] = (
        installment_level["AMT_INSTALMENT"] - installment_level["AMT_PAYMENT_TOTAL"]
    ).clip(lower=0)
    installment_level["INST_IS_UNDERPAYMENT"] = (
        installment_level["INST_PAYMENT_SHORTFALL"] > 0
    ).astype("int8")
    installment_level["INST_PAYMENT_SHORTFALL_RATE"] = safe_ratio(
        installment_level["INST_PAYMENT_SHORTFALL"], installment_level["AMT_INSTALMENT"]
    )
    installment_credit_features = (
        installment_level.groupby(["SK_ID_PREV", "SK_ID_CURR"])
        .agg(
            INST_PAYMENT_COUNT=("NUM_INSTALMENT_NUMBER", "count"),
            INST_LATE_PAYMENT_COUNT=("INST_IS_LATE_PAYMENT", "sum"),
            INST_UNDERPAYMENT_COUNT=("INST_IS_UNDERPAYMENT", "sum"),
            INST_MAX_DELAY_DAYS=("INST_PAYMENT_DELAY_DAYS", "max"),
            INST_MEAN_DELAY_DAYS=("INST_PAYMENT_DELAY_DAYS", "mean"),
            INST_MAX_SHORTFALL=("INST_PAYMENT_SHORTFALL", "max"),
            INST_MEAN_SHORTFALL=("INST_PAYMENT_SHORTFALL", "mean"),
            INST_MEAN_SHORTFALL_RATE=("INST_PAYMENT_SHORTFALL_RATE", "mean"),
            INST_MAX_SHORTFALL_RATE=("INST_PAYMENT_SHORTFALL_RATE", "max"),
            INST_TOTAL_REQUIRED_AMOUNT=("AMT_INSTALMENT", "sum"),
            INST_TOTAL_PAID_AMOUNT=("AMT_PAYMENT_TOTAL", "sum"),
            INST_MEAN_PAYMENT_PARTS=("PAYMENT_PARTS_COUNT", "mean"),
            INST_MAX_PAYMENT_PARTS=("PAYMENT_PARTS_COUNT", "max"),
        )
        .reset_index()
    )
    installment_credit_features["INST_LATE_PAYMENT_RATE"] = safe_ratio(
        installment_credit_features["INST_LATE_PAYMENT_COUNT"],
        installment_credit_features["INST_PAYMENT_COUNT"],
    )
    installment_credit_features["INST_UNDERPAYMENT_RATE"] = safe_ratio(
        installment_credit_features["INST_UNDERPAYMENT_COUNT"],
        installment_credit_features["INST_PAYMENT_COUNT"],
    )
    installment_credit_features["INST_TOTAL_PAYMENT_RATIO"] = safe_ratio(
        installment_credit_features["INST_TOTAL_PAID_AMOUNT"],
        installment_credit_features["INST_TOTAL_REQUIRED_AMOUNT"],
    )
    client_installment_features = (
        installment_credit_features.groupby("SK_ID_CURR")
        .agg(
            INST_CONTRACT_COUNT=("SK_ID_PREV", "count"),
            INST_TOTAL_PAYMENT_COUNT=("INST_PAYMENT_COUNT", "sum"),
            INST_TOTAL_LATE_PAYMENT_COUNT=("INST_LATE_PAYMENT_COUNT", "sum"),
            INST_TOTAL_UNDERPAYMENT_COUNT=("INST_UNDERPAYMENT_COUNT", "sum"),
            INST_MEAN_LATE_PAYMENT_RATE=("INST_LATE_PAYMENT_RATE", "mean"),
            INST_MAX_LATE_PAYMENT_RATE=("INST_LATE_PAYMENT_RATE", "max"),
            INST_MEAN_UNDERPAYMENT_RATE=("INST_UNDERPAYMENT_RATE", "mean"),
            INST_MAX_UNDERPAYMENT_RATE=("INST_UNDERPAYMENT_RATE", "max"),
            INST_MAX_DELAY_DAYS=("INST_MAX_DELAY_DAYS", "max"),
            INST_MEAN_DELAY_DAYS=("INST_MEAN_DELAY_DAYS", "mean"),
            INST_MAX_SHORTFALL=("INST_MAX_SHORTFALL", "max"),
            INST_MEAN_SHORTFALL_RATE=("INST_MEAN_SHORTFALL_RATE", "mean"),
            INST_MAX_SHORTFALL_RATE=("INST_MAX_SHORTFALL_RATE", "max"),
            INST_TOTAL_REQUIRED_AMOUNT=("INST_TOTAL_REQUIRED_AMOUNT", "sum"),
            INST_TOTAL_PAID_AMOUNT=("INST_TOTAL_PAID_AMOUNT", "sum"),
            INST_MEAN_PAYMENT_PARTS=("INST_MEAN_PAYMENT_PARTS", "mean"),
            INST_MAX_PAYMENT_PARTS=("INST_MAX_PAYMENT_PARTS", "max"),
        )
        .reset_index()
    )
    client_installment_features["INST_OVERALL_LATE_PAYMENT_RATE"] = safe_ratio(
        client_installment_features["INST_TOTAL_LATE_PAYMENT_COUNT"],
        client_installment_features["INST_TOTAL_PAYMENT_COUNT"],
    )
    client_installment_features["INST_OVERALL_UNDERPAYMENT_RATE"] = safe_ratio(
        client_installment_features["INST_TOTAL_UNDERPAYMENT_COUNT"],
        client_installment_features["INST_TOTAL_PAYMENT_COUNT"],
    )
    client_installment_features["INST_OVERALL_PAYMENT_RATIO"] = safe_ratio(
        client_installment_features["INST_TOTAL_PAID_AMOUNT"],
        client_installment_features["INST_TOTAL_REQUIRED_AMOUNT"],
    )
    return client_installment_features


def card_features(credit_card_balance) -> pd.DataFrame:
    credit_card_history = credit_card_balance.copy()
    credit_card_history["CC_LIMIT_UTILIZATION"] = safe_ratio(
        credit_card_history["AMT_BALANCE"],
        credit_card_history["AMT_CREDIT_LIMIT_ACTUAL"],
    )
    credit_card_history["CC_IS_DPD"] = (credit_card_history["SK_DPD"] > 0).astype(
        "int8"
    )
    credit_card_history["CC_IS_DPD_DEF"] = (
        credit_card_history["SK_DPD_DEF"] > 0
    ).astype("int8")
    credit_card_history["CC_IS_DPD_30_PLUS"] = (
        credit_card_history["SK_DPD"] >= 30
    ).astype("int8")
    credit_card_history["CC_PAYMENT_SHORTFALL"] = (
        credit_card_history["AMT_INST_MIN_REGULARITY"]
        - credit_card_history["AMT_PAYMENT_CURRENT"]
    ).clip(lower=0)
    credit_card_history["CC_IS_MIN_PAYMENT_MISSED"] = (
        credit_card_history["CC_PAYMENT_SHORTFALL"] > 0
    ).astype("int8")
    credit_card_history["CC_DRAWINGS_TO_LIMIT"] = safe_ratio(
        credit_card_history["AMT_DRAWINGS_CURRENT"],
        credit_card_history["AMT_CREDIT_LIMIT_ACTUAL"],
    )
    last_credit_card_records = (
        credit_card_history.sort_values(["SK_ID_PREV", "MONTHS_BALANCE"])
        .groupby("SK_ID_PREV", as_index=False)
        .tail(1)[
            [
                "SK_ID_PREV",
                "MONTHS_BALANCE",
                "AMT_BALANCE",
                "AMT_CREDIT_LIMIT_ACTUAL",
                "CC_LIMIT_UTILIZATION",
                "SK_DPD",
            ]
        ]
        .rename(
            columns={
                "MONTHS_BALANCE": "CC_LAST_MONTH",
                "AMT_BALANCE": "CC_LAST_BALANCE",
                "AMT_CREDIT_LIMIT_ACTUAL": "CC_LAST_CREDIT_LIMIT",
                "CC_LIMIT_UTILIZATION": "CC_LAST_LIMIT_UTILIZATION",
                "SK_DPD": "CC_LAST_DPD",
            }
        )
    )
    credit_card_features = (
        credit_card_history.groupby(["SK_ID_PREV", "SK_ID_CURR"])
        .agg(
            CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            CC_MEAN_BALANCE=("AMT_BALANCE", "mean"),
            CC_MAX_BALANCE=("AMT_BALANCE", "max"),
            CC_MEAN_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            CC_MAX_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "max"),
            CC_MEAN_LIMIT_UTILIZATION=("CC_LIMIT_UTILIZATION", "mean"),
            CC_MAX_LIMIT_UTILIZATION=("CC_LIMIT_UTILIZATION", "max"),
            CC_DPD_MONTHS=("CC_IS_DPD", "sum"),
            CC_DPD_DEF_MONTHS=("CC_IS_DPD_DEF", "sum"),
            CC_DPD_30_PLUS_MONTHS=("CC_IS_DPD_30_PLUS", "sum"),
            CC_MAX_DPD=("SK_DPD", "max"),
            CC_MEAN_DPD=("SK_DPD", "mean"),
            CC_MIN_PAYMENT_MISSED_MONTHS=("CC_IS_MIN_PAYMENT_MISSED", "sum"),
            CC_MEAN_PAYMENT_SHORTFALL=("CC_PAYMENT_SHORTFALL", "mean"),
            CC_MAX_PAYMENT_SHORTFALL=("CC_PAYMENT_SHORTFALL", "max"),
            CC_MEAN_DRAWINGS_TO_LIMIT=("CC_DRAWINGS_TO_LIMIT", "mean"),
            CC_MAX_DRAWINGS_TO_LIMIT=("CC_DRAWINGS_TO_LIMIT", "max"),
        )
        .reset_index()
    )
    credit_card_features["CC_DPD_RATE"] = safe_ratio(
        credit_card_features["CC_DPD_MONTHS"], credit_card_features["CC_MONTHS_COUNT"]
    )
    credit_card_features["CC_DPD_DEF_RATE"] = safe_ratio(
        credit_card_features["CC_DPD_DEF_MONTHS"],
        credit_card_features["CC_MONTHS_COUNT"],
    )
    credit_card_features["CC_DPD_30_PLUS_RATE"] = safe_ratio(
        credit_card_features["CC_DPD_30_PLUS_MONTHS"],
        credit_card_features["CC_MONTHS_COUNT"],
    )
    credit_card_features["CC_MIN_PAYMENT_MISSED_RATE"] = safe_ratio(
        credit_card_features["CC_MIN_PAYMENT_MISSED_MONTHS"],
        credit_card_features["CC_MONTHS_COUNT"],
    )
    credit_card_features = credit_card_features.merge(
        last_credit_card_records, on="SK_ID_PREV", how="left", validate="one_to_one"
    )
    client_credit_card_features = (
        credit_card_features.groupby("SK_ID_CURR")
        .agg(
            CC_CARD_COUNT=("SK_ID_PREV", "count"),
            CC_TOTAL_MONTHS=("CC_MONTHS_COUNT", "sum"),
            CC_MEAN_BALANCE=("CC_MEAN_BALANCE", "mean"),
            CC_MAX_BALANCE=("CC_MAX_BALANCE", "max"),
            CC_MEAN_CREDIT_LIMIT=("CC_MEAN_CREDIT_LIMIT", "mean"),
            CC_MAX_CREDIT_LIMIT=("CC_MAX_CREDIT_LIMIT", "max"),
            CC_MEAN_LIMIT_UTILIZATION=("CC_MEAN_LIMIT_UTILIZATION", "mean"),
            CC_MAX_LIMIT_UTILIZATION=("CC_MAX_LIMIT_UTILIZATION", "max"),
            CC_MEAN_DPD_RATE=("CC_DPD_RATE", "mean"),
            CC_MAX_DPD_RATE=("CC_DPD_RATE", "max"),
            CC_MEAN_DPD_DEF_RATE=("CC_DPD_DEF_RATE", "mean"),
            CC_MAX_DPD_30_PLUS_RATE=("CC_DPD_30_PLUS_RATE", "max"),
            CC_MAX_DPD=("CC_MAX_DPD", "max"),
            CC_MEAN_DPD=("CC_MEAN_DPD", "mean"),
            CC_MEAN_MIN_PAYMENT_MISSED_RATE=("CC_MIN_PAYMENT_MISSED_RATE", "mean"),
            CC_MAX_MIN_PAYMENT_MISSED_RATE=("CC_MIN_PAYMENT_MISSED_RATE", "max"),
            CC_MEAN_PAYMENT_SHORTFALL=("CC_MEAN_PAYMENT_SHORTFALL", "mean"),
            CC_MAX_PAYMENT_SHORTFALL=("CC_MAX_PAYMENT_SHORTFALL", "max"),
            CC_MEAN_DRAWINGS_TO_LIMIT=("CC_MEAN_DRAWINGS_TO_LIMIT", "mean"),
            CC_MAX_DRAWINGS_TO_LIMIT=("CC_MAX_DRAWINGS_TO_LIMIT", "max"),
            CC_LAST_BALANCE=("CC_LAST_BALANCE", "mean"),
            CC_LAST_CREDIT_LIMIT=("CC_LAST_CREDIT_LIMIT", "mean"),
            CC_LAST_LIMIT_UTILIZATION=("CC_LAST_LIMIT_UTILIZATION", "mean"),
            CC_LAST_DPD=("CC_LAST_DPD", "max"),
            CC_LAST_MONTH=("CC_LAST_MONTH", "max"),
        )
        .reset_index()
    )
    return client_credit_card_features


def previous_features(previous_application) -> pd.DataFrame:
    previous_history = previous_application.copy()
    previous_history["PREV_IS_APPROVED"] = (
        previous_history["NAME_CONTRACT_STATUS"] == "Approved"
    ).astype("int8")
    previous_history["PREV_IS_REFUSED"] = (
        previous_history["NAME_CONTRACT_STATUS"] == "Refused"
    ).astype("int8")
    previous_history["PREV_IS_CANCELED"] = (
        previous_history["NAME_CONTRACT_STATUS"] == "Canceled"
    ).astype("int8")
    previous_history["PREV_APPROVED_TO_APPLICATION_RATIO"] = safe_ratio(
        previous_history["AMT_CREDIT"], previous_history["AMT_APPLICATION"]
    )
    previous_history["PREV_DOWN_PAYMENT_RATE"] = safe_ratio(
        previous_history["AMT_DOWN_PAYMENT"], previous_history["AMT_CREDIT"]
    )
    previous_date_columns = [
        "DAYS_DECISION",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]
    previous_history[previous_date_columns] = previous_history[
        previous_date_columns
    ].replace(365243, np.nan)
    previous_history["PREV_DAYS_TO_FIRST_DUE"] = (
        previous_history["DAYS_FIRST_DUE"] - previous_history["DAYS_DECISION"]
    )
    previous_history["PREV_PLANNED_DURATION_DAYS"] = (
        previous_history["DAYS_LAST_DUE_1ST_VERSION"]
        - previous_history["DAYS_FIRST_DUE"]
    )
    client_previous_features = (
        previous_history.groupby("SK_ID_CURR")
        .agg(
            PREV_APPLICATION_COUNT=("SK_ID_PREV", "count"),
            PREV_APPROVED_COUNT=("PREV_IS_APPROVED", "sum"),
            PREV_REFUSED_COUNT=("PREV_IS_REFUSED", "sum"),
            PREV_CANCELED_COUNT=("PREV_IS_CANCELED", "sum"),
            PREV_MEAN_APPLICATION_AMOUNT=("AMT_APPLICATION", "mean"),
            PREV_MAX_APPLICATION_AMOUNT=("AMT_APPLICATION", "max"),
            PREV_MEAN_CREDIT_AMOUNT=("AMT_CREDIT", "mean"),
            PREV_MAX_CREDIT_AMOUNT=("AMT_CREDIT", "max"),
            PREV_MEAN_APPROVED_TO_APPLICATION_RATIO=(
                "PREV_APPROVED_TO_APPLICATION_RATIO",
                "mean",
            ),
            PREV_MIN_APPROVED_TO_APPLICATION_RATIO=(
                "PREV_APPROVED_TO_APPLICATION_RATIO",
                "min",
            ),
            PREV_MEAN_DOWN_PAYMENT_RATE=("PREV_DOWN_PAYMENT_RATE", "mean"),
            PREV_MEAN_DAYS_DECISION=("DAYS_DECISION", "mean"),
            PREV_LAST_DAYS_DECISION=("DAYS_DECISION", "max"),
            PREV_MEAN_DAYS_TO_FIRST_DUE=("PREV_DAYS_TO_FIRST_DUE", "mean"),
            PREV_MEAN_PLANNED_DURATION_DAYS=("PREV_PLANNED_DURATION_DAYS", "mean"),
        )
        .reset_index()
    )
    client_previous_features["PREV_APPROVED_RATE"] = safe_ratio(
        client_previous_features["PREV_APPROVED_COUNT"],
        client_previous_features["PREV_APPLICATION_COUNT"],
    )
    client_previous_features["PREV_REFUSED_RATE"] = safe_ratio(
        client_previous_features["PREV_REFUSED_COUNT"],
        client_previous_features["PREV_APPLICATION_COUNT"],
    )
    client_previous_features["PREV_CANCELED_RATE"] = safe_ratio(
        client_previous_features["PREV_CANCELED_COUNT"],
        client_previous_features["PREV_APPLICATION_COUNT"],
    )
    return client_previous_features


def merge_feature_tables(
    application: pd.DataFrame, features: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    result = application.copy()
    for prefix, feature_table in features.items():
        result = result.merge(
            feature_table, on=ID_COLUMN, how="left", validate="one_to_one"
        )
        feature_columns = [
            column for column in feature_table.columns if column != ID_COLUMN
        ]
        result[f"HAS_{prefix}_HISTORY"] = (
            result[feature_columns].notna().any(axis=1).astype("int8")
        )
    return result
