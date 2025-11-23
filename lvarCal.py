from typing import List

import numpy as np
import pandas as pd

from huihui import (
    attach_cHH_taylor,
    categorize_investors_by_edr,
    compute_all_hui_heubel,
    compute_LVaR_portfolio,
    compute_worst_case_edr_from_existing_LVaR,
    fit_quadratic_hh_zero_insignificant,
)
from mkt_day_end import get_historical_market_price, get_last_30_days_market_value_avg
from mkt_security import get_outstanding_shares_security_codes

# ================================================================
# PARAMETERS (policy-configurable)
# ================================================================
k = 0.015  # expected fractional slippage at 100% participation
gamma = 0.7  # market impact exponent (square-root law)
T = 5  # liquidation horizon (days)
alpha = 0.7  # blend between slippage and HH structural term
f = 0.6  # liquidation fraction


def lvrCal(csh: List[dict]):
    client_df, stock_df = quntileCal(csh)

    investor_summary_out = huihuiCal(client_df, stock_df)
    investor_summary_out = investor_summary_out.to_dict(orient="records")

    return {
        "status": 200,
        "message": "Success",
        "data": investor_summary_out,
    }


# quntileCal function to calculate quantile of the portfolio
def quntileCal(csh: List[dict]):
    # Get unique security codes from client holdings
    security_codes = [item["security_code"] for item in csh]
    security_codes.append("00DSEX")  # Add market index

    # Fetch historical market price data
    stock_data = get_historical_market_price(security_codes)
    stock_df = pd.DataFrame(stock_data)
    stock_df["date"] = pd.to_datetime(stock_df["date"], format="ISO8601")
    stock_df.set_index("date", inplace=True)
    stock_df = stock_df.reset_index()
    stock_df = stock_df.pivot(
        index="date", columns="security_code", values="market_price"
    )
    stock_returns = stock_df.pct_change(fill_method=None)
    client_df = pd.DataFrame(csh)

    adtv = get_last_30_days_market_value_avg(security_codes)
    adtv_df = pd.DataFrame(adtv, columns=["security_code", "ADTV"])
    adtv_df["ADTV"] = adtv_df["ADTV"].astype(float)
    adtv_df["ADTV"] = adtv_df["ADTV"] * 1000000

    # make todaysPrice as float
    client_df["todaysPrice"] = client_df["todaysPrice"].astype(float)
    client_df["quantity"] = client_df["quantity"].astype(float)
    client_df["equity"] = client_df["equity"].astype(float)
    client_df["liabilities"] = client_df["liabilities"].astype(float)
    client_df["total_market_value"] = client_df["total_market_value"].astype(float)
    # calculate total value
    client_df["market_value"] = client_df["quantity"] * client_df["todaysPrice"]
    client_df["liabalityRatio"] = (client_df["equity"] / client_df["liabilities"]) * 100

    # calculate weight of each security in the portfolio
    client_df["weight"] = client_df["market_value"] / client_df["total_market_value"]

    quntile_results = []
    for client in client_df["fintr_customer_id"].unique():
        client_data = client_df[client_df["fintr_customer_id"] == client]
        # STORE CLIENT ALL SECURITY CODE IN A LIST
        security_codes = client_data["security_code"].unique()
        investor_returns = stock_returns[security_codes]
        investor_returns = investor_returns.dropna()

        print(client_data.columns)

        # Check for duplicates and aggregate if needed
        if client_data["security_code"].duplicated().any():
            # Group by security_code and sum weights in case of duplicate holdings
            weights_series = client_data.groupby("security_code")["weight"].sum()
        else:
            # No duplicates, use directly
            weights_series = client_data.set_index("security_code")["weight"]

        weights = investor_returns.mul(weights_series, axis=1)
        port_returns = weights.sum(axis=1)

        # port_returns = weighted_returns.sum(axis=1)
        port_returns = port_returns.dropna()
        if len(port_returns) < 250:  # Minimum for quantile
            var_95 = 0
        else:
            # 95% VaR: 5% quantile of returns (loss tail)
            quantile_05 = np.quantile(port_returns, 0.05)
            # Get the first value of total_market_value (all rows have same value per client)
            total_portfolio_value = client_data["total_market_value"].iloc[0]
            var_95 = (
                -quantile_05 * total_portfolio_value
            )  # Positive loss value for 1-day VaR

        quntile_results.append(
            {
                "fintr_customer_id": client,
                "var_95": float(var_95),
            }
        )

    quntile_results = pd.DataFrame(quntile_results)
    client_df = client_df.merge(
        quntile_results[["fintr_customer_id", "var_95"]],
        on="fintr_customer_id",
        how="left",
    )
    client_df["var_95"] = client_df["var_95"].fillna(0)

    client_df = client_df.merge(adtv_df, on="security_code", how="left")
    return client_df, stock_data


# huihuiCal function to calculate huihui liquidity ratio
def huihuiCal(client_df, stock_df):
    print("huihuiCal test")
    client_df = client_df.rename(
        columns={
            "fintr_customer_id": "Investor Code",
            "full_name": "Investor_Name",
            "security_code": "Instrument Code",
            "quantity": "TotalStock",
            "avgcost": "AvgCost",
            "todaysPrice": "Market_Price",
            "total_cost": "Total_Cost",
            "market_value": "TotalMarketValue",
            "liabilities": "Asset/Liabilities",
            "equity": "Equity",
            "liabalityRatio": "Equity_Debt_Ratio",
            "var_95": "VaR_95",
        }
    )
    price_df = pd.DataFrame(stock_df)
    price_df = price_df.rename(
        columns={
            "security_code": "Scrip",
            "ad_close": "Adj_Close",
            "ad_open": "Adj_Open",
            "ad_high": "Adj_High",
            "ad_low": "Adj_Low",
            "ad_volume": "Adj_Volume",
        }
    )
    # print(client_df.head())
    # print(price_df)

    outstanding_shares = get_outstanding_shares_security_codes()
    outstanding_shares_df = pd.DataFrame(
        outstanding_shares,
        columns=[
            "scrip",
            "outstanding_shares_mn",
            "floating_shares_mn",
            "floating_percent",
        ],
    )
    floating_shares = dict(
        zip(outstanding_shares_df["scrip"], outstanding_shares_df["floating_shares_mn"])
    )

    hui_huebel_df = compute_all_hui_heubel(price_df, floating_shares)
    fitted_hh_df = fit_quadratic_hh_zero_insignificant(hui_huebel_df)
    # Attach structural liquidity term (Taylor-based)
    merged_df = attach_cHH_taylor(
        client_df, fitted_hh_df, hui_huebel_df, floating_shares
    )
    merged_df, investor_LVaR = compute_LVaR_portfolio(merged_df, f)

    merged_df_enriched, investor_summary = compute_worst_case_edr_from_existing_LVaR(
        merged_df
    )
    investor_summary_out, merged_df_out, cat_counts = categorize_investors_by_edr(
        investor_summary, merged_df
    )

    # Replace NaN, inf, -inf with None (null in JSON) or 0 for safe JSON serialization
    investor_summary_out = investor_summary_out.replace([np.nan, np.inf, -np.inf], None)

    return investor_summary_out
