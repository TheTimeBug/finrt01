import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict
from statsmodels.tsa.seasonal import seasonal_decompose
# ================================================================
# PARAMETERS (policy-configurable)
# ================================================================
k = 0.015      # expected fractional slippage at 100% participation
gamma = 0.7   # market impact exponent (square-root law)
T = 5         # liquidation horizon (days)
alpha = 0.7   # blend between slippage and HH structural term
f = 0.6       # liquidation fraction

outstanding_shares = {'BATBC':540,'BEACONPHAR':231,'BEXIMCO':943.24,'BRACBANK':1769.71,'BSC':152.53,'BSCPLC':187.04,'BSRMLTD':298.58,
                      'BXPHARMA':446.11,'CITYBANK':1347.08,'DBH':198.89,'DELTALIFE':123.75,'EBL':1358.14,'ETL':183.74,'FBFIF':776.10,
                      'FIRSTSBANK':1208.14,'GPHISPAT':483.88,'IDLC':415.70,'IFIC':1922.09,'ITC':128.59,'JAMUNAOIL':110.42,'KBPPWBIL':98.08,
                      'KOHINOOR':37.07,'LANKABAFIN':538.84,'LHB':1161.37,'MERCANBANK':1106.58,'MJLBD':316.75,'MPETROLEUM':108.22,
                      'OLYMPIC':199.94,'PADMAOIL':98.23,'POWERGRID':913.81, 'PRIMEBANK': 1132.28, 'PUBALIBANK':1156.83,'RENATA':114.70,
                      'SANDHANINS':109.70,'SOUTHEASTB':1337.40,'SQURPHARMA':886.45,'TRUSTB1MF':303.59, 'UNIQUEHRL':294.40,'WALTONHIL':302.93,
                      '1STPRIMFMF':20.00,'GP':1350.40,'ROBI':5237.93}
def remove_outliers_iqr(series, multiplier=3):
    """
    Remove outliers from a Pandas Series based on the IQR method,
    keeping original NaNs unchanged.

    Parameters:
        series (pd.Series): Input Series.
        multiplier (float): IQR multiplier (default 3).

    Returns:
        pd.Series: Series with outliers replaced by NaN, original NaNs preserved.
    """
    # Calculate bounds ignoring NaNs
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Create a mask for valid (non-outlier) values
    is_valid = (series >= lower_bound) & (series <= upper_bound)

    # Return a series: keep values where valid, set outliers to NaN
    return series.where(is_valid | series.isna())

def hui_heubel_liquidity_ratio(ohlcv_data: pd.DataFrame, shares_outstanding: float, window: int = 5):
    """
    Calculate the Hui-Heubel Liquidity Ratio over non-overlapping weekly windows from daily OHLCV data.

    The Hui-Heubel Liquidity Ratio is a measure of market liquidity, defined as:
        L_HH = ((P_max - P_min) / P_min) / (V / S)
    where:
        - P_max: Maximum adjusted high price within the window
        - P_min: Minimum adjusted low price within the window
        - V: Total adjusted volume over the window (in number of shares)
        - S: Shares outstanding (in millions)

    This function computes the ratio using rolling windows of trading days (default 5, i.e., a week),
    stepping forward by the window size to avoid overlap.

    Parameters:
        ohlcv_data (pd.DataFrame): DataFrame with columns ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume'],
                                   indexed by DateTime. Volume must be in number of shares.
        shares_outstanding (float): Total shares outstanding (in millions).
        window (int): Number of trading days per window (default is 5 for weekly).

    Returns:
        pd.DataFrame: DataFrame indexed by the last date of each window, with columns:
                      - 'Hui_Heubel_Liquidity_Ratio'
                      - 'Price_Range'
                      - 'Turnover'

    Raises:
        TypeError: If `ohlcv_data` is not a DataFrame.
        ValueError: If required columns are missing, data contains invalid values, or input parameters are inappropriate.
    """
    # Input validation
    if not isinstance(ohlcv_data, pd.DataFrame):
        raise TypeError("ohlcv_data must be a pandas DataFrame")
        # Convert index to DateTime if not already
    if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
        try:
            ohlcv_data = ohlcv_data.copy()  # Avoid modifying the input DataFrame
            ohlcv_data.index = pd.to_datetime(ohlcv_data.index)
        except (ValueError, TypeError) as e:
            raise ValueError("Index could not be converted to DateTime. Ensure index contains valid date-like values.") from e
    required_columns = ['Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume']
    if not all(col in ohlcv_data.columns for col in required_columns):
        raise ValueError(f"ohlcv_data must contain columns: {required_columns}")
    if shares_outstanding <= 0:
        raise ValueError("shares_outstanding must be positive")
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if window > len(ohlcv_data):
        raise ValueError("window size exceeds data length")
    if (ohlcv_data[['Adj_High', 'Adj_Low', 'Adj_Close']] <= 0).any().any():
        raise ValueError("OHLCV data contains non-positive prices")
    if (ohlcv_data['Adj_Volume'] < 0).any():
        raise ValueError("OHLCV data contains negative volumes")

    # Resample data to weekly periods (window trading days)
    # Use rolling window of 'window' days, stepping by 'window' to avoid overlap
    ratios = []
    price_ranges = []
    turnovers = []   
    dates = []

    for start in range(0, len(ohlcv_data) - window + 1, window):
        end = start + window
        window_data = ohlcv_data.iloc[start:end]

        # Ensure the window has enough data
        if len(window_data) < window:
            print(f"Window limit not met at {window_data.index[-1]}")
            continue

        # Calculate weekly metrics
        p_max = window_data['Adj_High'].max()  # Weekly high
        p_min = window_data['Adj_Low'].min()   # Weekly low
        volume = window_data['Adj_Volume'].sum()  # Weekly volume (in shares)

        # Calculate numerator: (P_max - P_min) / P_min
        if p_min == 0:  # Avoid division by zero
            price_range = np.nan
            ratio = np.nan
            turnover = np.nan
        else:
            price_range = (p_max - p_min) / p_min

            # Calculate denominator: V / S (turnover ratio)
            if volume == 0:  # Avoid division by zero
                ratio = np.nan
            else:
                turnover = volume / (shares_outstanding*1000000)

                # Calculate Hui-Heubel Liquidity Ratio
                if turnover == 0:  # Avoid division by zero
                    ratio = np.nan
                else:
                    ratio = price_range / turnover

        ratios.append(ratio)
        price_ranges.append(price_range)
        turnovers.append(turnover)
        dates.append(window_data.index[-1])  # Last date of the window

    # Create output DataFrame
    result = pd.DataFrame({
        'Hui_Heubel_Liquidity_Ratio': ratios,
        'Price_Range': price_ranges,
        'Turnover': turnovers
    }, index=dates)

    # Remove infinite values
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def variance_ratio_series(price_series: pd.Series, window: int = 63, step: int = 20):
    """
    Calculate the ratio of the variance of 20-day returns to the variance of 1-day returns,
    for both percentage returns and logarithmic returns, using a rolling 1-year window,
    updating every ~20 trading days.

    Parameters:
        price_series (pd.Series): Series of adjusted close prices with DateTime index.
        window (int): Size of rolling window in trading days (default 63).
        step (int): Step size in trading days (default 20).

    Returns:
        tuple:
            - pd.Series: Variance ratios for percentage returns, sampled every 'step' days.
            - pd.Series: Variance ratios for logarithmic returns, sampled every 'step' days.
            - pd.Series: 20-day percentage returns.
            - pd.Series: 1-day percentage returns.
            - pd.Series: 20-day logarithmic returns.
            - pd.Series: 1-day logarithmic returns.
    """
    # Calculate 1-day and 20-day returns
    daily_returns = price_series.pct_change()
    twenty_day_returns = price_series.pct_change(20)
  
    # Calculate logarithmic returns
    log_prices = np.log(price_series)
    daily_log_returns = log_prices.diff()  # 1-day log returns
    twenty_day_log_returns = log_prices.diff(20)  # 20-day log returns
    # Debugging output
    # print(f"Daily returns length: {len(daily_returns)}, 20-day returns length: {len(twenty_day_returns)}")
    # print(f"Daily log returns length: {len(daily_log_returns)}, 20-day log returns length: {len(twenty_day_log_returns)}")
    # print("Sample 20-day percentage returns:\n", twenty_day_returns.head())
    # print("Sample 20-day log returns:\n", twenty_day_log_returns.head())
    # Initialize lists for variance ratios
    perc_var_ratio_list = []
    log_var_ratio_list = []
    dates_list = []
    # Rolling calculation
    for start in range(0, len(price_series) - window, 5):
        end = start + window
        if end > len(price_series):
            break
        
        daily_window = daily_returns.iloc[start:end]
        twenty_window = twenty_day_returns.iloc[start:end]

        var_1d = np.nanvar(daily_window, ddof=1)  # sample variance (unbiased)
        var_20d = np.nanvar(twenty_window, ddof=1)
         # Logarithmic returns variances
        daily_log_window = daily_log_returns.iloc[start:end]
        twenty_log_window = twenty_day_log_returns.iloc[start:end]
        var_1d_log = np.nanvar(daily_log_window, ddof=1)
        var_20d_log = np.nanvar(twenty_log_window, ddof=1)
        if var_1d != 0:  # avoid division by zero
            perc_ratio = var_20d /(step* var_1d)
        else:
            perc_ratio = np.nan
        # Calculate logarithmic return variance ratio
        if var_1d_log != 0:  # Avoid division by zero
            log_ratio = var_20d_log / (step * var_1d_log)  # Use 20 as the return period
        else:
            log_ratio = np.nan
        perc_var_ratio_list.append(perc_ratio)
        log_var_ratio_list.append(log_ratio)
        dates_list.append(price_series.index[end-1])  # Record date at window end

    # Create output Series
    perc_var_ratio_series = pd.Series(perc_var_ratio_list, index=dates_list, name="Percentage_Variance_Ratio")
    log_var_ratio_series = pd.Series(log_var_ratio_list, index=dates_list, name="Log_Variance_Ratio")

    return (perc_var_ratio_series, log_var_ratio_series,
            twenty_day_returns, daily_returns,
            twenty_day_log_returns, daily_log_returns)

def average_monthly_volume(df: pd.DataFrame, scrip_name: str) -> pd.Series:
    """
    Calculates the average monthly volume (in BDT) for each calendar month
    (i.e., average total February volume across all Februarys, and so on)
    for a particular scrip.

    Parameters:
        df (pd.DataFrame): Full dataset containing 'Date', 'Scrip', and 'Volume_BDT' columns.
        scrip_name (str): The specific scrip to filter and analyze.

    Returns:
        pd.Series: A Series indexed by month name (e.g., 'January', 'February', ...) 
                   with the average monthly total volume for that calendar month.
    """
    # Filter for the selected scrip and required columns
    df_scrip = df[df['Scrip'] == scrip_name].copy()
    
    # Ensure datetime index
    # df_scrip['Date'] = pd.to_datetime(df_scrip['Date'])
    # df_scrip.set_index('Date', inplace=True)

    # Resample monthly and sum Volume_BDT per month
    monthly_volume = df_scrip['Volume_BDT'].resample('ME').sum()

    # Extract month names from datetime index
    monthly_volume_df = monthly_volume.to_frame(name='Monthly_Volume')
    monthly_volume_df['Month'] = monthly_volume_df.index.month_name()

    # Group by month and compute average of total monthly volumes
    average_volume_by_month = monthly_volume_df.groupby('Month')['Monthly_Volume'].mean()

    # Optional: sort by calendar order
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    average_volume_by_month = average_volume_by_month.reindex(month_order)

    return average_volume_by_month

def analyze_seasonality(df, column='Volume_BDT'):
    """
    Analyze seasonality in the specified column after applying log transformation.

    Parameters:
        df (pd.DataFrame): DataFrame with DateTime index and target column.
        column (str): The column on which to perform the seasonal decomposition.

    Returns:
        decomposition (DecomposeResult): Object with seasonal, trend, resid, and observed components.
    """
    # Step 1: Resample to monthly frequency
    monthly_data = df[column].resample('ME').mean()

    # Step 2: Interpolate and fill missing values
    monthly_data = monthly_data.interpolate(method='linear').bfill().ffill()

    # Step 3: Log transformation
    monthly_data = monthly_data.replace(0, np.nan).dropna()
    monthly_data = np.log(monthly_data)

    # Step 4: Seasonal decomposition (additive in log space)
    decomposition = seasonal_decompose(monthly_data, model='additive', period=12)

    return decomposition



def fix_last_nan(col):
    if pd.isna(col.iloc[-1]):
        col.iloc[-1] = col.ffill().iloc[-1]
    return col

def compute_all_hui_heubel(price_df: pd.DataFrame, floating_shares: Dict[str, float]) -> pd.DataFrame:
    """
    Compute weekly Hui–Heubel Liquidity Ratio and turnover for each stock in price_df.

    Parameters
    ----------
    price_df : pd.DataFrame
        Must contain columns ['Scrip', 'Adj_High', 'Adj_Low', 'Adj_Volume'] and DateTime index.
    floating_shares : dict
        Mapping of scrip -> floating shares in millions.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns:
        ['Scrip', 'Hui_Heubel_Liquidity_Ratio', 'Price_Range', 'Turnover']
    """
    all_results = []

    for scrip, stock_data in price_df.groupby("Scrip"):
        if scrip not in floating_shares:
            print(f"Skipping {scrip}: missing floating shares info.")
            continue

        shares = floating_shares[scrip]  # in millions
        try:
            # --- Compute HH ratio (your function) ---
            result = hui_heubel_liquidity_ratio(stock_data, shares)

            # --- Remove outliers (your function) ---
            # result_clean = result.apply(lambda col: remove_outliers_iqr(col, multiplier=6))
            result_clean = result.apply(lambda col: remove_outliers_iqr(col, multiplier=6)).apply(fix_last_nan) #<- New Update
            # if scrip in ['MIRAKHTER','SILCOPHL','ROBI']:
            #     print(f"Post-cleaning Hui-huebel Ratio for {scrip}:\n{result_clean()}")
            result_clean["Scrip"] = scrip
            all_results.append(result_clean)

        except Exception as e:
            print(f"Error processing {scrip}: {e}")
            continue

    if not all_results:
        raise ValueError("No valid results produced. Check your data and floating shares mapping.")

    all_results_df = pd.concat(all_results)
    all_results_df = all_results_df.reset_index().rename(columns={"index": "Date"})
    return all_results_df

def fit_quadratic_hh_zero_insignificant(all_results_df: pd.DataFrame, alpha: float = 0.05, min_obs: int = 8) -> pd.DataFrame:
    """
    Fit HH = a + b*q + c*q² per scrip.
    If a coefficient's p-value > alpha, set that coefficient to 0.
    Recompute R² after adjustment.
    Record which ones were not significant.
    """
    results = []

    for scrip, group in all_results_df.groupby("Scrip"):
        df = group.dropna(subset=["Hui_Heubel_Liquidity_Ratio", "Turnover"]).copy()
        df = df[df["Turnover"] > 0]

        if len(df) < min_obs:
            continue

        q = df["Turnover"].values
        hh = df["Hui_Heubel_Liquidity_Ratio"].values

        # Base OLS regression
        X = np.column_stack([np.ones_like(q), q, q**2])
        model = sm.OLS(hh, X).fit()

        a, b, c = model.params
        p_a, p_b, p_c = model.pvalues

        # Identify insignificant coefficients
        insignificants = []
        if p_a > alpha:
            a = 0
            insignificants.append("a")
        if p_b > alpha:
            b = 0
            insignificants.append("b")
        if p_c > alpha:
            c = 0
            insignificants.append("c")

        # Create updated significance note
        significance_comment = (
            "All significant" if not insignificants else f"{', '.join(insignificants)} not significant"
        )

        # --- Recompute fitted values and R² using adjusted coefficients ---
        hh_pred_adj = a + b * q + c * q**2
        ss_res_adj = np.sum((hh - hh_pred_adj) ** 2)
        ss_tot = np.sum((hh - np.mean(hh)) ** 2)
        R2_adj = 1 - ss_res_adj / ss_tot if ss_tot > 0 else np.nan

        # --- Derivatives at median turnover ---
        q_med = np.median(q)
        hh_prime_med = b + 2 * c * q_med
        hh_dd = 2 * c

        results.append({
            "Scrip": scrip,
            "a": a,
            "b": b,
            "c": c,
            "R2": R2_adj,            # adjusted R² after zeroing
            "n_obs": len(df),
            "p_a": p_a,
            "p_b": p_b,
            "p_c": p_c,
            "q_median": q_med,
            "HH_prime_med": hh_prime_med,
            "HH_dd": hh_dd,
            "Significance": significance_comment
        })

    return pd.DataFrame(results)

def attach_cHH_taylor(merged_df: pd.DataFrame,fitted_hh_df: pd.DataFrame,hui_huebel_df: pd.DataFrame,floating_shares: dict) -> pd.DataFrame:
    """
    Compute structural liquidity cost cHH_i using second-order Taylor expansion:
    
        HH(q_i + f_i) ≈ HH(q_i) + HH'(q_i)f_i + 0.5*HH''(q_i)f_i^2
        cHH_i = HH(q_i + f_i) * f_i

    where:
        f_i = TotalStock / FloatingShares
        HH(q_i) = latest observed Hui–Heubel liquidity ratio from hui_huebel_df
        HH'(q_i) = b + 2c*q_i
        HH''(q_i) = 2c
    """
    df = merged_df.copy()
    hh_fit = fitted_hh_df.rename(columns={"Scrip": "Instrument Code"})
    hh_obs = hui_huebel_df.rename(columns={"Scrip": "Instrument Code"})

    # --- Merge coefficients (a,b,c) ---
    df = df.merge(hh_fit[["Instrument Code", "a", "b", "c"]], on="Instrument Code", how="left")

    # --- Merge latest observed HH(q_i) ---
    hh_obs_latest = hh_obs.sort_values(["Instrument Code", "Date"]).groupby("Instrument Code").tail(1)
    df = df.merge(hh_obs_latest[["Instrument Code", "Hui_Heubel_Liquidity_Ratio", "Turnover"]],
                  on="Instrument Code", how="left")
    df.rename(columns={"Hui_Heubel_Liquidity_Ratio": "HH_qi_obs"}, inplace=True)

    # --- Floating shares mapping ---
    df["FloatingShares_mn"] = df["Instrument Code"].map(floating_shares)
    df["TotalStock_mn"] = df["TotalStock"] / 1e6

    # --- Investor’s fractional position ---
    df["f_i"] = df["TotalStock_mn"] / df["FloatingShares_mn"]

    # --- Local derivatives of HH(q) ---
    df["HH_prime_qi"] = df["b"] + 2 * df["c"] * df["Turnover"]
    df["HH_dd_qi"] = 2 * df["c"]

    # --- Taylor expansion: HH(q_i + f_i) ≈ HH(q_i) + HH'(q_i)*f_i + 0.5*HH''(q_i)*f_i^2 ---
    df["HH_qi_plus_fi"] = (
        df["HH_qi_obs"] + df["HH_prime_qi"] * df["f_i"] + 0.5 * df["HH_dd_qi"] * (df["f_i"] ** 2)
    )

    # --- Structural liquidity cost ---
    df["cHH_i"] = df["HH_qi_plus_fi"] * df["f_i"]

    # --- Clean up governance ---
    df.loc[df["cHH_i"] < 0, "cHH_i"] = np.nan
    df["cHH_i"] = df["cHH_i"].clip(upper=1)
    # --- Drop intermediate columns ---
    drop_cols = ["a", "b", "c", "HH_prime_qi", "HH_dd_qi", "HH_qi_plus_fi"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    return df


# ================================================================
# STEP 2 — Define Slippage and Liquidity Cost (LC_i)
# ================================================================
def compute_slippage(N_i, ADTV_i, T, k, gamma):
    """Temporary slippage fraction."""
    if ADTV_i is None or np.isnan(ADTV_i) or ADTV_i <= 0:
        return np.nan
    ratio = N_i / (ADTV_i * T)
    return k * (ratio ** gamma)

def compute_liquidity_cost(row):
    """Compute LC_i (min rule). Requires columns TotalMarketValue, ADTV, cHH_i."""
    slip_i = compute_slippage(row["TotalMarketValue"], row["ADTV"], T, k, gamma)
    cHH_i = row.get("cHH_i", 0.015)  # fallback example; replace with actual cHH_i
    LC_i = min(1, alpha * slip_i + (1 - alpha) * cHH_i)
    return LC_i, slip_i
# ================================================================
# STEP 3 — Compute LVaR per investor
# ================================================================
def compute_LVaR_portfolio(merged_df: pd.DataFrame, f: float) -> pd.DataFrame:
    """
    Compute per-position and per-investor LVaR and add as new columns.
    Requires columns: Investor Code, VaR_95, TotalMarketValue, ADTV, cHH_i.
    """
    df = merged_df.copy()
    
    # Compute LC_i and slip_i
    df[["LC_i", "Slip_i"]] = df.apply(lambda r: pd.Series(compute_liquidity_cost(r)), axis=1)
    
    # Compute liquidity loss in BDT
    df["Liquidity_Loss_BDT"] = df["LC_i"] * df["TotalMarketValue"]
    
    investor_LVaR = (
        df.groupby("Investor Code")
        .agg({
            "VaR_95": "first",
            "Liquidity_Loss_BDT": "sum"
        })
        .assign(LVaR_95=lambda x: np.where(
            (x["VaR_95"].fillna(0) > 1),
            x["VaR_95"] + f * x["Liquidity_Loss_BDT"],
            0
        ))
    )


    
    # Map LVaR back to original df
    df["LVaR_95"] = df["Investor Code"].map(investor_LVaR["LVaR_95"])
    
    return df, investor_LVaR


def compute_worst_case_edr_from_existing_LVaR(merged_df: pd.DataFrame,
                                              lvar_col: str = "LVaR_95",
                                              asset_liab_col: str = "Asset/Liabilities"):
    """
    Use existing per-row LVaR_95 to compute investor-level worst-case Market Value,
    Worst-case Equity, Worst-case EDR (Equity-to-Debt Ratio), and Default_Flag,
    then map results back into merged_df.

    Assumptions:
      - merged_df contains columns:
          ['Investor Code', 'TotalMarketValue', 'Equity', asset_liab_col, lvar_col]
      - lvar_col contains per-investor LVaR already (duplicated on each row),
        but function will take the investor-level first() to be safe.
      - Debt is derived from negative Asset/Liabilities entries (sum of negatives -> positive debt).
    Returns:
      df_out: merged_df with new columns (investor-level mapped):
        - Investor_TotalMarketValue
        - Investor_LVaR_95
        - Investor_Equity
        - Investor_Debt
        - WorstCase_MarketValue
        - WorstCase_Equity
        - WorstCase_EDR
        - Current_EDR
        - Default_Flag
        - LVaR_to_VaR_Ratio
      investor_summary: aggregated per-investor table with same columns.
    """

    df = merged_df.copy()

    # helper: return positive debt magnitude
    def first_negative_as_positive(s):
        if s.dropna().empty:
            return 0.0
        neg_values = s[s < 0]
        if neg_values.empty:
            return 0.0
        return float(-neg_values.iloc[0])


    # aggregate per investor
    agg = df.groupby("Investor Code").agg({
        "TotalMarketValue": "sum",
        lvar_col: "first",             # take first available LVaR (assumes same for investor)
        "VaR_95": "first",
        "Equity": "first",
        asset_liab_col: lambda x: first_negative_as_positive(x) # take first available liability (assumes same for investor)
    }).rename(columns={
        "TotalMarketValue": "Investor_TotalMarketValue",
        lvar_col: "Investor_LVaR_95",
        "VaR_95": "Investor_VaR_95",
        "Equity": "Investor_Equity",
        asset_liab_col: "Investor_Debt"
    }).reset_index()

    # Ensure numeric types
    for col in ["Investor_TotalMarketValue", "Investor_LVaR_95", "Investor_VaR_95", "Investor_Equity", "Investor_Debt"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce").fillna(0.0)

    # Worst-case market value & equity after applying investor LVaR (if LVaR is already 0 when VaR==0, it's fine)
    agg["WorstCase_MarketValue"] = agg["Investor_TotalMarketValue"] - agg["Investor_LVaR_95"]
    agg["WorstCase_Equity"] = agg["Investor_Equity"] - agg["Investor_LVaR_95"]

    # Worst-case EDR = WorstCase_Equity / Investor_Debt (only when debt > 0), else NaN
    agg["WorstCase_EDR"] = np.where(
        agg["Investor_Debt"] > 0,
        agg["WorstCase_Equity"] / agg["Investor_Debt"],
        np.nan
    )

    # Current (pre-shock) EDR = Investor_Equity / Investor_Debt
    agg["Current_EDR"] = np.where(
        agg["Investor_Debt"] > 0,
        agg["Investor_Equity"] / agg["Investor_Debt"],
        np.nan
    )

    # Default flag if worst-case equity <= 0
    agg["Default_Flag"] = agg["WorstCase_Equity"] <= 0

    # LVaR to VaR ratio for diagnostics (NaN if VaR==0)
    agg["LVaR_to_VaR_Ratio"] = np.where(
        agg["Investor_VaR_95"] > 0,
        agg["Investor_LVaR_95"] / agg["Investor_VaR_95"],
        np.nan
    )

    # Map back to merged_df
    map_cols = [
        "Investor_TotalMarketValue", "Investor_LVaR_95", "Investor_VaR_95",
        "Investor_Equity", "Investor_Debt",
        "WorstCase_MarketValue", "WorstCase_Equity", "WorstCase_EDR",
        "Current_EDR", "Default_Flag", "LVaR_to_VaR_Ratio"
    ]
    maps = {col: dict(zip(agg["Investor Code"], agg[col])) for col in map_cols}
    for col in map_cols:
        df[col] = df["Investor Code"].map(maps[col])

    # Return enriched df and investor summary
    return df, agg



def categorize_investors_by_edr(investor_summary: pd.DataFrame, merged_df: pd.DataFrame = None):
    """
    Classify investors into risk categories based only on existing
    WorstCase_EDR (WE) and Current_EDR (CE) columns.
    
    Parameters
    ----------
    investor_summary : pd.DataFrame
        Must contain ['Investor Code', 'WorstCase_EDR', 'Current_EDR'].
    merged_df : pd.DataFrame, optional
        If provided, adds corresponding 'EDR_Category' column per investor.

    Returns
    -------
    investor_summary_out : pd.DataFrame
        investor_summary with 'EDR_Category' column.
    merged_df_out : pd.DataFrame or None
        merged_df with mapped category column if provided.
    category_counts : pd.DataFrame
        counts of investors in each category.
    """
    inv = investor_summary.copy()

    def classify_row(we, ce):
        if pd.isna(we) or pd.isna(ce):
            return "No Debt / Insufficient Data"

        if we < 0:
            return "Category 1: WE<0 & CE<0" if ce < 0 else "Category 2: WE<0 & CE>=0"

        if 0 <= we < 0.25:
            return "Category 3: 0<=WE<0.25 & CE<0.5" if ce < 0.5 else "category 3b: 0<=WE<0.25 & CE>=0.5"

        if 0.25 <= we < 0.5:
            if ce < 0.5:
                return "Category 4: 0.25<=WE<0.5 & CE<0.5"
            elif 0.5<=ce<0.75:
                return "Category 5: 0.25<=WE<0.5 & 0.5<=CE<0.75"
            else:
                return "Category 5b: 0.25<=WE<0.5 & CE>=0.75"

        if 0.5 <= we < 0.75:
            return "Category 6: 0.5<=WE<0.75 & CE<0.75" if ce < 0.75 else "Category 7: 0.5<=WE<0.75 & CE>=0.75"

        if 0.75 <= we < 1:
            return "Category 8: 0.75<=WE<1 & 0.75<=CE < 1" if 0.75 <= ce < 1 else "Category 9: 0.75<=WE<1 & CE>=1"

        if we > 1:
            return "Category 10: WE>1 & CE>1"

        return "Uncategorized"


    # --- Apply classification ---
    inv["EDR_Category"] = inv.apply(lambda r: classify_row(r["WorstCase_EDR"], r["Current_EDR"]), axis=1)

    # --- Category counts ---
    category_counts = inv["EDR_Category"].value_counts(dropna=False).reset_index()
    category_counts.columns = ["EDR_Category", "count"]

    # --- Map to merged_df if provided ---
    merged_out = None
    if merged_df is not None:
        merged_out = merged_df.copy()
        cat_map = dict(zip(inv["Investor Code"], inv["EDR_Category"]))
        merged_out["EDR_Category"] = merged_out["Investor Code"].map(cat_map)

    return inv, merged_out, category_counts


# Example usage:
# investor_summary_out, merged_df_out, cat_counts = categorize_investors_by_edr(investor_summary, merged_df)
# print(cat_counts)
