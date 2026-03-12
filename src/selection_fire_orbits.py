import pandas as pd

def infer_platform_from_sysindex(x: str) -> str:
    if pd.isna(x):
        return "UNK"
    s = str(x).strip().upper()
    if "_" in s:
        pref = s.split("_", 1)[0]
        if pref in ("S1A", "S1B"):
            return pref
    if "S1A" in s:
        return "S1A"
    if "S1B" in s:
        return "S1B"
    return "UNK"

def platform_priority(series: pd.Series, prefer: str = "S1A") -> pd.Series:
    return (series != prefer).astype(int)

def tiebreak_sort(cand: pd.DataFrame, col_slice: str, col_sysindex: str,
                  slice_rank: dict, prefer_platform_name: str = "S1A") -> pd.DataFrame:
    cand = cand.copy()
    cand["_p_plat"] = platform_priority(cand["platform"], prefer_platform_name)
    cand["_slice_num"] = pd.to_numeric(cand[col_slice], errors="coerce")
    cand["_p_slice_rank"] = cand["_slice_num"].map(slice_rank)
    cand["_p_slice_rank"] = cand["_p_slice_rank"].fillna(10**9).astype("int64")
    return cand.sort_values(
        ["_p_plat", "_p_slice_rank", col_sysindex],
        ascending=[True, True, True]
    )

def choose_same_date(meta_df, fire_date, col_img_date, col_slice, col_sysindex, slice_rank, prefer_platform):
    cand = meta_df.loc[meta_df["img_date"] == fire_date]
    if cand.empty:
        return None
    cand = tiebreak_sort(cand, col_slice, col_sysindex, slice_rank, prefer_platform)
    return cand.iloc[0]

def choose_post_window(meta_df, fire_date, col_slice, col_sysindex, slice_rank,
                       prefer_platform, post_min_days, post_max_days):
    start = fire_date + pd.Timedelta(days=post_min_days)
    end = fire_date + pd.Timedelta(days=post_max_days)
    cand = meta_df.loc[(meta_df["img_date"] >= start) & (meta_df["img_date"] <= end)]
    if cand.empty:
        return None
    max_date = cand["img_date"].max()
    cand = cand.loc[cand["img_date"] == max_date]
    cand = tiebreak_sort(cand, col_slice, col_sysindex, slice_rank, prefer_platform)
    return cand.iloc[0]

def choose_pre_window(meta_df, fire_date, col_slice, col_sysindex, slice_rank,
                      prefer_platform, pre_min_days, pre_target_days):
    start = fire_date - pd.Timedelta(days=pre_min_days)
    target = fire_date - pd.Timedelta(days=pre_target_days)
    cand = meta_df.loc[(meta_df["img_date"] >= start) & (meta_df["img_date"] <= target)].copy()
    if cand.empty:
        return None
    cand["abs_diff_days"] = (cand["img_date"] - target).abs().dt.days
    min_abs = cand["abs_diff_days"].min()
    cand = cand.loc[cand["abs_diff_days"] == min_abs]
    cand = tiebreak_sort(cand, col_slice, col_sysindex, slice_rank, prefer_platform)
    return cand.iloc[0]