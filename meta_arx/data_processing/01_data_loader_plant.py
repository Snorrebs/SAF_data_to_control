from typing import Optional
from pathlib import Path
import duckdb
import os
import pandas as pd


from datetime import datetime, timedelta
import re


"""
 Build a training-ready, downsampled CSV from the large PI-data.csv export using DuckDB.

Overview
--------
Reads the large PI-data CSV, renames raw column headers to readable names (see RAW_TO_READABLE),
resamples to fixed time buckets, aggregates values, and keeps rows when controls change or at a
regular anchor interval—then writes a compact training CSV.

Requirements
------------
- duckdb (pip install duckdb)
- pandas 

Input assumptions
-----------------
- The input CSV has a timestamp column whose raw name is mapped to TIME_COL ("timestamp")
  via RAW_TO_READABLE ("Datetime" -> "timestamp")
- Control/target columns in CONTROL_COLS and TARGET_COL must also be present in RAW_TO_READABLE
- Transformer tap columns are integers, others are numeric (float)
- Add any new collumns you want included to RAW_TO_READABLE, the refrenes can be found in PI-data.pdf in the wacker zip

What the code does
------------
1) Renames columns using RAW_TO_READABLE
2) Buckets timestamps: floor to "resample_seconds" (e.g 30s)
3) Aggregates per bucket:
   - Numeric (controls + target): AVG
   - Tap/step columns ("Transformer step"): MIN (last known within bucket if monotone)
4) Change detection (window over bucketed time):
   - Keep a row if any numeric control changed by > "change_eps" since previous bucket,
   - OR any tap/step value changed,
   - OR it's an anchor row where (row_number % keep_every_n == 0)
   Note: The row number (and thus anchoring) resets each calendar month
5) Writes CSV ordered by timestamp with columns: [timestamp] + CONTROL_COLS + [TARGET_COL]

Time range
----------
- "start / end" are date strings "YYYY-MM-DD"
- "end" is exclusive (keeps rows where timestamp < end)
- If start/end are omitted, they're inferred from the input CSV's time column

Parameters
----------
input_csv : str   Path to source CSV, to get the path on windows simply go to the file and rightclick press "copy as path", obs! if it fails swap "/" with "\"
output_csv: str   Path to write trimmed CSV
start/end : str   Date bounds ("YYYY-MM-DD"). "end" is exclusive
resample_seconds: int  Bucket size in seconds (default 30)
keep_every_n    : int  Always keep every Nth bucket (monthly reset)
change_eps      : float Minimal numeric delta to treat as a change

Output
------
A CSV at "output_csv" with headers:
[timestamp, <controls...>, Furnace power_MW], deduplicated/resampled

Common pitfalls
---------------
- KeyError/empty output: ensure RAW_TO_READABLE includes *all* required raw columns
- Wrong time window: remember `end` is exclusive (e.g., end="2022-07-08" covers up to 07-07 23:59:59)
- Sparse data kept: increase `keep_every_n` or lower "change_eps"
- Large files: script scans month-by-month to limit memory, but it still reads the file per month

Example
-------
make_training_csv_chunked(
    input_csv=r"C:/path/to/PI-data.csv",
    output_csv=r"C:/path/to/training.csv",
    start="2022-07-01",
    end="2022-07-08",
    resample_seconds=30,
    keep_every_n=60,
    change_eps=1e-3,
)
"""

print("[running file]", __file__)
print("[abs path]    ", Path(__file__).resolve())

# Raw to readable names
RAW_TO_READABLE = {
    "Datetime": "timestamp",

    # -------- Furnace-level electricals (A) --------
    r"\\ZMUCPI01\V1903T830EU102.1 Effekt":        "Furnace_power_MW",
    r"\\ZMUCPI01\V1903T870EU104 CalcResTot":      "Tot_Resistance_mOhm",

    # -------- Furnace C3 (A/B important) --------
    r"\\ZMUCPI01\V1903T830EU101.1 C3":            "Furnace_C3",

    # -------- Electrode holder positions (A) --------
    r"\\ZMUCPI01\V1903T830EG104.1 GI":            "El1_pos_m",
    r"\\ZMUCPI01\V1903T830EG204.1 GI":            "El2_pos_m",
    r"\\ZMUCPI01\V1903T830EG304.1 GI":            "El3_pos_m",

    # -------- Phase voltages (A) --------
    r"\\ZMUCPI01\V1903T870EE976.1 UL1N":          "UL1N_V",
    r"\\ZMUCPI01\V1903T870EE976.1 UL2N":          "UL2N_V",
    r"\\ZMUCPI01\V1903T870EE976.1 UL3N":          "UL3N_V",

    # -------- Electrode currents (A) --------
    r"\\ZMUCPI01\V1903T830EU100.1 Strøm":         "El1_kA",
    r"\\ZMUCPI01\V1903T830EU200.1 Strøm":         "El2_kA",
    r"\\ZMUCPI01\V1903T830EU300.1 Strøm":         "El3_kA",

    # -------- Electrode resistances per phase (A, important for future) --------
    r"\\ZMUCPI01\V1903T830EU100.1 Resistans":     "El1_Resistance_mOhm",
    r"\\ZMUCPI01\V1903T830EU200.1 Resistans":     "El2_Resistance_mOhm",
    r"\\ZMUCPI01\V1903T830EU300.1 Resistans":     "El3_Resistance_mOhm",

    # -------- Transformer taps (A) --------
    r"\\ZMUCPI01\V1903T830EU102.1 TCA":           "Tap_A",
    r"\\ZMUCPI01\V1903T830EU102.1 TCB":           "Tap_B",
    r"\\ZMUCPI01\V1903T830EU102.1 TCC":           "Tap_C",

    # -------- Setpoints (trimmed B: only furnace + El1) --------
    r"\\ZMUCPI01\V1903T830EU102.1 Effekt SP":     "Furnace_power_MW_SP",
    r"\\ZMUCPI01\V1903T830EU101.1 C3 SP":         "Furnace_C3_SP",

    r"\\ZMUCPI01\V1903T830EU100.1 Resistans SP":  "El1_Resistance_SP_mOhm",
    r"\\ZMUCPI01\V1903T830EU100.1 Strøm SP":      "El1_kA_SP",

    # -------- El1 regulation modes (trimmed B) --------
    r"\\ZMUCPI01\V1903T830EU100.1 StrømModus":         "El1_current_mode",
    r"\\ZMUCPI01\V1903T830EU100.1 ResistansModus":     "El1_resistance_mode",
}

CONTROL_COLS = [
    # --- ARX-critical signals ---
    "UL1N_V", "UL2N_V", "UL3N_V",        # phase voltages
    "El1_pos_m", "El2_pos_m", "El3_pos_m",
    "El1_kA", "El2_kA", "El3_kA",
    "Tap_A", "Tap_B", "Tap_C",

    # --- Furnace-level context ---
    "Furnace_power_MW",
    "Furnace_C3",

    # --- Electrode-level context (for later models/diagnostics) ---
    "El1_Resistance_mOhm", "El2_Resistance_mOhm", "El3_Resistance_mOhm",

    # --- Setpoints (trimmed B) ---
    "Furnace_power_MW_SP", "Furnace_C3_SP",
    "El1_Resistance_SP_mOhm", "El1_kA_SP",

    # --- Regulation modes (trimmed B) ---
    "El1_current_mode", "El1_resistance_mode",
]

TARGET_COL = "Tot_Resistance_mOhm"
TIME_COL   = "timestamp"

TARGET_COL = "Tot_Resistance_mOhm"
TIME_COL   = "timestamp"


TARGET_COL = "Tot_Resistance_mOhm"
TIME_COL   = "timestamp"



def _sanitize(name: str) -> str:
    return re.sub(r"\W+", "_", name)

def _month_range(start: datetime, end: datetime):
    # Ensure both are tz-aware and in the same zone (default: Europe/Oslo if missing)
    import pandas as pd
    OSLO = "Europe/Oslo"

    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    if s.tz is None: s = s.tz_localize(OSLO)
    if e.tz is None: e = e.tz_localize(OSLO)

    cur = s.normalize().replace(day=1)
    stop = e.normalize().replace(day=1)
    while cur <= stop:
        nxt = (cur + pd.offsets.MonthBegin(1))
        # yield tz-aware Python datetimes
        yield cur.to_pydatetime(), nxt.to_pydatetime()
        cur = nxt

def make_training_csv_chunked(
    input_csv: str,
    output_csv: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    resample_seconds: int = 30,
    keep_every_n: int = 60,
    change_eps: float = 1e-3,
) -> str:
    
    input_csv = Path(input_csv).as_posix()
    output_csv = Path(output_csv).as_posix()

    # find raw time column
    raw_time = None
    for raw, friendly in RAW_TO_READABLE.items():
        if friendly == TIME_COL:
            raw_time = raw
            break
    if raw_time is None:
        raise ValueError("Time column mapping not found in RAW_TO_READABLE.")

    con = duckdb.connect()

    # derive time bounds if missing
    if not start or not end:
        mm = con.execute(f"""
            SELECT
              min(CAST("{raw_time}" AS TIMESTAMP)) AS tmin,
              max(CAST("{raw_time}" AS TIMESTAMP)) AS tmax
            FROM read_csv_auto('{input_csv}', SAMPLE_SIZE=-1, columns={{"{raw_time}":"VARCHAR"}})
            WHERE "{raw_time}" IS NOT NULL
        """).fetchdf()
        tmin, tmax = mm.loc[0, "tmin"], mm.loc[0, "tmax"]
        if pd.isna(tmin) or pd.isna(tmax):
            raise ValueError("Could find time range from input CSV.")
        start = (pd.to_datetime(start) if start else pd.to_datetime(tmin)).strftime("%Y-%m-%d")
        end   = (pd.to_datetime(end)   if end   else pd.to_datetime(tmax)).strftime("%Y-%m-%d")

    start_dt = pd.to_datetime(start).to_pydatetime()
    end_dt   = pd.to_datetime(end).to_pydatetime()

    # SELECT aliases for reader friendly headers
    select_aliases = [f'"{raw}" AS "{friendly}"' for raw, friendly in RAW_TO_READABLE.items()]

    # decide which controls are taps vs numeric
    tap_cols = [c for c in CONTROL_COLS if c.lower().startswith("tap")]
    num_cols = [c for c in CONTROL_COLS if c not in tap_cols]

    # target is numeric too
    num_cols_with_target = num_cols + [TARGET_COL]

    # Use REPLACE to accept comma decimals, TRY_CAST so non-numeric cells become NULL
    def cast_double_expr(col: str) -> str:
        return f'TRY_CAST(REPLACE("{col}", \',\', \'.\') AS DOUBLE) AS "{col}"'
    def cast_int_expr(col: str) -> str:
        return f'TRY_CAST("{col}" AS INTEGER) AS "{col}"'

    cast_selects = []
    for c in num_cols_with_target:
        cast_selects.append(cast_double_expr(c))
    for c in tap_cols:
        cast_selects.append(cast_int_expr(c))
    cast_selects_sql = ",\n            ".join(cast_selects)

    # aggregation step
    bucket = f"to_timestamp(floor(epoch({TIME_COL})/{resample_seconds})*{resample_seconds})"
    avg_select = ",\n        ".join([f'avg("{c}") AS "{c}"' for c in num_cols_with_target])
    tap_select = ",\n        ".join([f'min("{c}") AS "{c}"' for c in tap_cols]) if tap_cols else ""
    tap_chunk  = (",\n        " + tap_select) if tap_select else ""

    # deltas to detect movement
    delta_aliases = []
    for c in CONTROL_COLS:
        safe = _sanitize(c)
        if c in tap_cols:
            delta_aliases.append(f'("{c}" <> lag("{c}") OVER w) AS chg_{safe}')
        else:
            delta_aliases.append(f'abs("{c}" - lag("{c}") OVER w) AS d_{safe}')
    deltas_sql = ",\n            ".join(delta_aliases)

    delta_pred = " OR ".join(
        [f"coalesce(d_{_sanitize(c)}, 0) > {change_eps}" for c in num_cols]
    ) or "FALSE"
    tap_pred = " OR ".join(
        [f"coalesce(chg_{_sanitize(c)}, FALSE)" for c in tap_cols]
    ) or "FALSE"
    keep_anchor = f"(rn % {keep_every_n} = 0)"


    control_cols_joined = ", ".join([f'"{c}"' for c in CONTROL_COLS])
    final_cols = [TIME_COL] + CONTROL_COLS + [TARGET_COL]
    final_select = ", ".join([f'"{c}"' for c in final_cols])

    # run month by month
    # partwise using SQL
    header_written = False
    total_rows = 0
    print(f"[chunked] building -> {output_csv}")
    for m_start, m_next in _month_range(start_dt, end_dt):
        # keep these as tz-aware timestamps
        chunk_start = max(m_start, start_dt)
        chunk_end   = min(m_next,  end_dt)

        # make ISO strings for SQL literals (do NOT overwrite chunk_start/end)
        cs = pd.Timestamp(chunk_start).isoformat()
        ce = pd.Timestamp(chunk_end).isoformat()

        where_time = (
            f'WHERE CAST("{raw_time}" AS TIMESTAMPTZ) >= TIMESTAMPTZ \'{cs}\' '
            f'AND   CAST("{raw_time}" AS TIMESTAMPTZ) <  TIMESTAMPTZ \'{ce}\''
        )

        sql_body = f"""
        WITH src AS (
        SELECT
            {", ".join(select_aliases)}
        FROM read_csv_auto('{input_csv}', SAMPLE_SIZE=-1)
        {where_time}
        ),
        t AS (
        SELECT
            CAST({TIME_COL} AS TIMESTAMPTZ) AS {TIME_COL},
            {cast_selects_sql}
        FROM src
        WHERE {TIME_COL} IS NOT NULL
        ),
        ds AS (
        SELECT
            {bucket} AS bucket_ts,
            {avg_select}{tap_chunk}
        FROM t
        GROUP BY 1
        ),
        chg AS (
        SELECT
            *,
            {deltas_sql}
        FROM ds
        WINDOW w AS (ORDER BY bucket_ts)
        ),
        kept AS (
        SELECT
            bucket_ts AS "{TIME_COL}",
            {control_cols_joined},
            "{TARGET_COL}"
        FROM (
            SELECT
            chg.*,
            row_number() OVER (ORDER BY bucket_ts) AS rn
            FROM chg
        )
        WHERE {tap_pred} OR {delta_pred} OR {keep_anchor}
        )
        SELECT {final_select}
        FROM kept
        ORDER BY "{TIME_COL}"
        """

        nrows = con.execute(f"SELECT count(*) FROM ({sql_body})").fetchone()[0]
        print(f"[chunked] {chunk_start:%Y-%m} rows={nrows}")

        if nrows and not header_written:
            con.execute(f"COPY ({sql_body}) TO '{output_csv}' (HEADER, DELIMITER ',');")
            header_written = True
        elif nrows:
            con.execute(f"COPY ({sql_body}) TO '{output_csv}' (HEADER false, DELIMITER ',');")

        total_rows += int(nrows)

    if total_rows == 0:
        print("[chunked] no rows produced in the selected range.")
    else:
        rng = con.execute(
            f'SELECT min("{TIME_COL}"), max("{TIME_COL}") FROM read_csv_auto(\'{output_csv}\', SAMPLE_SIZE=-1)'
        ).fetchone()
        print(f"[chunked] done: {output_csv} | rows={total_rows} | range: {rng[0]} -> {rng[1]}")

    return output_csv



if __name__ == "__main__":

    # Replace valus with your values
    make_training_csv_chunked(
        input_csv=r"meta_arx/data/raw/PI-data.csv",
        output_csv=r"meta_arx/data/1s_data_from_plant/0702_0703_1s_new.csv",
        start="2022-07-07 00:01:01+02:00",
        end="2022-07-07 23:59:01+02:00",
        resample_seconds=1,
        keep_every_n=300,
        change_eps=5e-4,
    )