"""Build application-level Home Credit repayment features with bounded memory.

Uses pandas, numpy, and Python's built-in sqlite3 module. No new dependency.

Example in Jupyter (put this file beside your notebook):
    from home_credit_repayment_features import build_repayment_features
    repayment_features_full = build_repayment_features(
        installment_files[0],
        application_train['SK_ID_CURR'],
        work_dir=Path(installment_files[0]).parent,
    )

The CSV is read completely into a temporary, on-disk SQLite database, selecting
only the requested application IDs. Grouping happens AFTER ingestion, so payments
split between input chunks are reconciled together. The temporary database is
removed on completion or a handled exception. An interrupted process may leave
its uniquely named temporary directory; it can be removed once that process stops.

Grouping: SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_NUMBER. Conflicting schedule
versions/dates/amounts require review; card versions and entries due today or
later are excluded from assessed-history percentages. Missing required schedule
fields, missing payment dates, or missing/negative eligible payment amounts also
prevent assessment. Payments dated after zero do not contribute to amount sums.

This reproduces the earlier classify_instalment STATUS logic for finite numeric
data. It does not calculate COVERAGE_GAP_DAYS or reconstruct contractual schedules.
For non-negative payments, cumulative coverage by a date is equivalent to the
total paid by that date reaching the scheduled amount. No payment sorting or
first-coverage date is needed for the selected percentage features.

Review rules inspect all recorded versions and dates, just as the exploratory
classifier did. The dataset does not provide a complete event-time audit trail
for schedule revisions; the day-zero filter is not proof of production as-of
availability. TARGET is never accepted or used by this feature builder.
"""

from pathlib import Path
import sqlite3
import tempfile
import time

import numpy as np
import pandas as pd


PAYMENT_COLUMNS = [
    "SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_NUMBER",
    "NUM_INSTALMENT_VERSION", "DAYS_INSTALMENT", "AMT_INSTALMENT",
    "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT",
]

COUNT_COLUMNS = [
    "RECORDED_INSTALMENT_GROUPS", "COVERED_ON_TIME", "COVERED_LATE",
    "NOT_FULLY_COVERED", "REVIEW_GROUPS", "CARD_GROUPS",
    "DUE_TODAY_OR_LATER_GROUPS",
]


# MIN/MAX and missing flags detect conflicts across the COMPLETE group. Required
# field checks precede the card classification, matching the original function.
SUMMARY_SQL = """
WITH instalment_groups AS (
    SELECT
        SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_NUMBER,
        MAX(CASE WHEN SK_ID_PREV IS NULL
                   OR NUM_INSTALMENT_NUMBER IS NULL
                   OR NUM_INSTALMENT_VERSION IS NULL
                   OR DAYS_INSTALMENT IS NULL
                   OR AMT_INSTALMENT IS NULL
                 THEN 1 ELSE 0 END) AS MISSING_REQUIRED,
        MIN(NUM_INSTALMENT_VERSION) AS VERSION_MIN,
        MAX(NUM_INSTALMENT_VERSION) AS VERSION_MAX,
        MIN(DAYS_INSTALMENT) AS DUE_MIN,
        MAX(DAYS_INSTALMENT) AS DUE_MAX,
        MIN(AMT_INSTALMENT) AS SCHEDULED_MIN,
        MAX(AMT_INSTALMENT) AS SCHEDULED_MAX,
        MAX(CASE WHEN DAYS_ENTRY_PAYMENT IS NULL
                 THEN 1 ELSE 0 END) AS MISSING_PAYMENT_DATE,
        MAX(CASE WHEN DAYS_ENTRY_PAYMENT <= 0
                  AND (AMT_PAYMENT IS NULL OR AMT_PAYMENT < 0)
                 THEN 1 ELSE 0 END) AS BAD_ELIGIBLE_PAYMENT,
        SUM(CASE WHEN DAYS_ENTRY_PAYMENT <= 0
                 THEN 1 ELSE 0 END) AS ELIGIBLE_RECORDS,
        SUM(CASE WHEN DAYS_ENTRY_PAYMENT <= 0
                  AND DAYS_ENTRY_PAYMENT <= DAYS_INSTALMENT
                 THEN 1 ELSE 0 END) AS RECORDS_BY_DUE,
        SUM(CASE WHEN DAYS_ENTRY_PAYMENT <= 0
                 THEN COALESCE(AMT_PAYMENT, 0) ELSE 0 END) AS PAID_BY_CUTOFF,
        SUM(CASE WHEN DAYS_ENTRY_PAYMENT <= 0
                  AND DAYS_ENTRY_PAYMENT <= DAYS_INSTALMENT
                 THEN COALESCE(AMT_PAYMENT, 0) ELSE 0 END) AS PAID_BY_DUE
    FROM payments
    GROUP BY SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_NUMBER
),
classified AS (
    SELECT SK_ID_CURR,
        CASE
            WHEN MISSING_REQUIRED = 1 OR VERSION_MIN <> VERSION_MAX
                THEN 'Review required'
            WHEN VERSION_MIN = 0
                THEN 'Credit card: separate analysis'
            WHEN VERSION_MIN < 0 OR DUE_MIN <> DUE_MAX
                OR SCHEDULED_MIN <> SCHEDULED_MAX OR SCHEDULED_MIN <= 0
                THEN 'Review required'
            WHEN DUE_MIN >= 0
                THEN 'Due today or later'
            WHEN MISSING_PAYMENT_DATE = 1 OR BAD_ELIGIBLE_PAYMENT = 1
                THEN 'Review required'
            WHEN ELIGIBLE_RECORDS = 0
                OR PAID_BY_CUTOFF < SCHEDULED_MIN - :tolerance
                THEN 'Not fully covered by cutoff'
            WHEN RECORDS_BY_DUE > 0
                AND PAID_BY_DUE >= SCHEDULED_MIN - :tolerance
                THEN 'Covered on time or early'
            ELSE 'Covered late'
        END AS STATUS
    FROM instalment_groups
)
SELECT SK_ID_CURR,
    COUNT(*) AS RECORDED_INSTALMENT_GROUPS,
    SUM(CASE WHEN STATUS = 'Covered on time or early'
             THEN 1 ELSE 0 END) AS COVERED_ON_TIME,
    SUM(CASE WHEN STATUS = 'Covered late'
             THEN 1 ELSE 0 END) AS COVERED_LATE,
    SUM(CASE WHEN STATUS = 'Not fully covered by cutoff'
             THEN 1 ELSE 0 END) AS NOT_FULLY_COVERED,
    SUM(CASE WHEN STATUS = 'Review required'
             THEN 1 ELSE 0 END) AS REVIEW_GROUPS,
    SUM(CASE WHEN STATUS = 'Credit card: separate analysis'
             THEN 1 ELSE 0 END) AS CARD_GROUPS,
    SUM(CASE WHEN STATUS = 'Due today or later'
             THEN 1 ELSE 0 END) AS DUE_TODAY_OR_LATER_GROUPS
FROM classified
GROUP BY SK_ID_CURR
"""


def build_repayment_features(
    csv_path,
    application_ids,
    work_dir=None,
    chunksize=300_000,
    tolerance=0.000001,
    verbose=True,
):
    """Return counts/percentages for every requested ID, in requested order.

    Counts are zero for IDs without matching records; percentages remain NaN
    when no entries are assessable. Card/review/future exclusions are retained
    as separate audit counts. All counts concern recorded instalment groups.

    work_dir is a local folder with space for an on-disk working database and
    SQLite's intermediate grouping data. Runtime/disk use depend on the input
    and computer; progress messages appear during ingestion and aggregation.
    This always rebuilds from the CSV; it never reuses an unverified cache.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    if not isinstance(chunksize, int) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    ids = pd.Series(list(application_ids), name="SK_ID_CURR")
    if ids.empty or ids.isna().any():
        raise ValueError("application_ids must be non-empty and non-missing")
    ids = pd.to_numeric(ids, errors="raise")
    if not np.isfinite(ids.to_numpy(dtype=float)).all() or (ids % 1 != 0).any():
        raise ValueError("application_ids must contain finite integers")
    ids = ids.astype("int64")
    if ids.duplicated().any():
        raise ValueError("application_ids must be unique")
    requested = ids.to_frame()
    selected_ids = set(ids.tolist())

    if work_dir is None:
        work_dir = csv_path.parent
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    rows_scanned = rows_kept = 0
    with tempfile.TemporaryDirectory(
        prefix="home_credit_repayments_", dir=work_dir
    ) as temporary_dir:
        database_path = Path(temporary_dir) / "payments.sqlite"
        connection = sqlite3.connect(str(database_path))
        try:
            connection.execute("PRAGMA temp_store = FILE")
            connection.execute("PRAGMA cache_size = -64000")
            connection.execute("""
                CREATE TABLE payments (
                    SK_ID_CURR INTEGER,
                    SK_ID_PREV REAL,
                    NUM_INSTALMENT_NUMBER REAL,
                    NUM_INSTALMENT_VERSION REAL,
                    DAYS_INSTALMENT REAL,
                    AMT_INSTALMENT REAL,
                    DAYS_ENTRY_PAYMENT REAL,
                    AMT_PAYMENT REAL
                )
            """)
            for chunk_number, chunk in enumerate(
                pd.read_csv(
                    csv_path, usecols=PAYMENT_COLUMNS, chunksize=chunksize,
                    dtype={column: "float64" for column in PAYMENT_COLUMNS},
                ),
                start=1,
            ):
                rows_scanned += len(chunk)
                matched = chunk.loc[
                    chunk["SK_ID_CURR"].isin(selected_ids), PAYMENT_COLUMNS
                ].copy()
                if not matched.empty:
                    if np.isinf(matched.to_numpy()).any():
                        raise ValueError("Infinite input values require inspection")
                    matched["SK_ID_CURR"] = matched["SK_ID_CURR"].astype("int64")
                    matched.to_sql(
                        "payments", connection, if_exists="append",
                        index=False, chunksize=10_000,
                    )
                    rows_kept += len(matched)
                if verbose:
                    print(
                        f"Chunk {chunk_number}: scanned {rows_scanned:,}; "
                        f"matched {rows_kept:,} payment records",
                        flush=True,
                    )
            if verbose:
                print(
                    "Reconciling complete instalment groups on disk. "
                    "This stage may take longer and emits no per-group progress.",
                    flush=True,
                )
            summary = pd.read_sql_query(
                SUMMARY_SQL, connection, params={"tolerance": float(tolerance)}
            )
        finally:
            connection.close()

    # SQLite SUM returns zero/NULL according to group existence. Only absent
    # groups are filled here; ambiguous groups were explicitly counted as review.
    result = requested.merge(
        summary, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False
    )
    result[COUNT_COLUMNS] = (
        result[COUNT_COLUMNS].astype("float64").fillna(0).astype("int64")
    )
    result["ASSESSED_DUE_ENTRIES"] = result[
        ["COVERED_ON_TIME", "COVERED_LATE", "NOT_FULLY_COVERED"]
    ].sum(axis=1)
    result["EXCLUDED_ENTRIES"] = (
        result["RECORDED_INSTALMENT_GROUPS"] - result["ASSESSED_DUE_ENTRIES"]
    )
    denominator = result["ASSESSED_DUE_ENTRIES"].where(
        result["ASSESSED_DUE_ENTRIES"] > 0
    )
    result["NOT_COVERED_ON_TIME_PCT"] = 100 * (
        result["COVERED_LATE"] + result["NOT_FULLY_COVERED"]
    ) / denominator
    result["INCOMPLETE_ENTRY_PCT"] = 100 * result["NOT_FULLY_COVERED"] / denominator

    if not np.array_equal(result["SK_ID_CURR"].to_numpy(), ids.to_numpy()):
        raise AssertionError("Requested application order was not preserved")
    if not result["EXCLUDED_ENTRIES"].equals(
        result[["REVIEW_GROUPS", "CARD_GROUPS", "DUE_TODAY_OR_LATER_GROUPS"]]
        .sum(axis=1)
    ):
        raise AssertionError("Exclusion counts do not reconcile")
    if verbose:
        print(
            f"Finished: {len(result):,} applications, "
            f"{result['RECORDED_INSTALMENT_GROUPS'].sum():,} instalment groups, "
            f"{time.monotonic() - started:.1f} seconds.",
            flush=True,
        )
    return result
