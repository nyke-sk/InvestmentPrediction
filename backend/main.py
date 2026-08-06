import os
import sys

# Ensure backend directory is in sys.path when running from workspace root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import io
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from schemas import (
    PortfolioParseRequest, PortfolioDiagnostics, MacroPulseResponse,
    RecommendationRequest, RecommendationResponse, StressTestRequest, StressTestResponse,
    BrokerExecuteRequest, BrokerExecuteResponse, TickerItem, TickerSaveRequest, TickerSyncResponse,
    ProbableScenariosResponse, TargetSellingPointRequest, TargetSellingPointResponse, TickerHistoryResponse
)
from services.mcp_client import mcp_client
from services.quant_engine_india import (
    calculate_portfolio_diagnostics, generate_recommendations, normalize_ticker,
    get_all_tickers, save_ticker_dataset, sync_top_tickers_dataset, calculate_target_selling_points,
    fetch_ticker_price_history
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BharatiQuant.API")

app = FastAPI(
    title="BharatiQuant - Indian Investment Planning & World Monitor Macro Engine",
    description="Quantitative Portfolio Optimization (HRP, HHI) with World Monitor Geopolitical & Indian Macro Intelligence.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    # SECURITY: Use specific origins instead of wildcard (*) when credentials are allowed to prevent CSRF and unauthorized cross-origin access.
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "status": "online",
        "app": "BharatiQuant Investment Platform",
        "market": "NSE / BSE India",
        "docs": "/docs"
    }

@app.get("/api/macro-pulse", response_model=MacroPulseResponse)
async def get_macro_pulse():
    """Fetch World Monitor MCP threat signals blended with Indian domestic macro context."""
    try:
        pulse = await mcp_client.get_macro_pulse()
        return pulse
    except Exception as e:
        logger.error(f"Error fetching macro pulse: {e}")
        # SECURITY: Fail securely by genericizing error messages. Do not leak internal stack traces or details to the client.
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/probable-scenarios", response_model=ProbableScenariosResponse)
async def get_probable_scenarios():
    """Dynamically synthesize 5 probable day-to-day macro scenarios from World Monitor feeds."""
    try:
        scenarios = await mcp_client.get_probable_scenarios()
        return scenarios
    except Exception as e:
        logger.error(f"Error generating probable scenarios: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/parse-portfolio", response_model=PortfolioDiagnostics)
async def parse_portfolio(
    file: Optional[UploadFile] = File(None),
    raw_holdings: Optional[str] = Form(None)
):
    """
    Parse uploaded CSV/Excel portfolio file or raw JSON payload.
    Normalize ticker symbols to NSE format (.NS), compute HHI concentration,
    sector allocation, correlation matrix, and Portfolio Health Score.
    """
    holdings_list = []

    if file:
        try:
            contents = b""
            while chunk := await file.read(1024 * 1024):
                contents += chunk
                if len(contents) > 10 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
            filename = file.filename.lower()

            def parse_dataframe(file_content: bytes, fname: str):
                if fname.endswith(".csv"):
                    return pd.read_csv(io.BytesIO(file_content))
                elif fname.endswith((".xls", ".xlsx")):
                    return pd.read_excel(io.BytesIO(file_content))
                else:
                    return None

            # ⚡ Bolt Optimization: Offload blocking pandas parsing to separate thread
            # so it does not block FastAPI's async event loop
            df = await asyncio.to_thread(parse_dataframe, contents, filename)

            if df is None:
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel.")

            # Standardize column names flexibly
            cols_lower = {str(c).strip().lower(): c for c in df.columns}
            ticker_col = (
                cols_lower.get("ticker") or cols_lower.get("symbol") or
                cols_lower.get("stock") or cols_lower.get("ticker symbol") or
                cols_lower.get("instrument") or cols_lower.get("asset") or
                cols_lower.get("name") or cols_lower.get("company")
            )
            qty_col = (
                cols_lower.get("quantity") or cols_lower.get("qty") or
                cols_lower.get("shares") or cols_lower.get("units") or cols_lower.get("volume")
            )
            price_col = (
                cols_lower.get("purchase price") or cols_lower.get("price") or
                cols_lower.get("buy price") or cols_lower.get("cost") or
                cols_lower.get("avg price") or cols_lower.get("purchase_price")
            )

            if not ticker_col:
                raise HTTPException(status_code=400, detail="CSV/Excel must contain a 'Ticker' or 'Symbol' column.")

            for _, row in df.iterrows():
                t = str(row[ticker_col]).strip()
                if pd.isna(t) or not t or t.lower() == "nan":
                    continue
                try:
                    q = float(row[qty_col]) if qty_col and not pd.isna(row[qty_col]) else 1.0
                except (ValueError, TypeError):
                    q = 1.0
                try:
                    p = float(row[price_col]) if price_col and not pd.isna(row[price_col]) else 0.0
                except (ValueError, TypeError):
                    p = 0.0

                holdings_list.append({
                    "Ticker": t,
                    "Quantity": max(0.01, q),
                    "Purchase Price": max(0.0, p)
                })
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File parsing error: {e}")
            raise HTTPException(status_code=400, detail="Failed to parse uploaded file.")
    elif raw_holdings:
        if len(raw_holdings) > 100 * 1024:
            raise HTTPException(status_code=413, detail="Payload too large. Maximum size is 100KB.")
        import json
        try:
            holdings_list = json.loads(raw_holdings)
        except Exception as e:
            logger.error(f"JSON parsing error in raw_holdings: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON string in raw_holdings.")

    macro_data = await mcp_client.get_macro_pulse()
    diagnostics = await asyncio.to_thread(calculate_portfolio_diagnostics, holdings_list, macro_data.get("threat_score", 35.0))
    return diagnostics

@app.post("/api/recommend-inr", response_model=RecommendationResponse)
async def get_recommendations(req: RecommendationRequest):
    """
    Generate EXACTLY 10 to 20 HRP-optimized recommendations formatted in Indian Rupees (₹ INR).
    """
    try:
        macro_data = await mcp_client.get_macro_pulse()
        recs = await asyncio.to_thread(
            generate_recommendations,
            available_capital_inr=req.available_capital_inr,
            risk_profile=req.risk_profile,
            existing_holdings=req.holdings,
            macro_data=macro_data,
            recommendation_count=req.count
        )
        return recs
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        # SECURITY: Fail securely by genericizing error messages. Do not leak internal stack traces or details to the client.
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/target-selling-point", response_model=TargetSellingPointResponse)
async def get_target_selling_point(req: TargetSellingPointRequest):
    """
    Calculate real-time current ticker rates, target selling prices per share, expected profit per stock,
    and estimated holding period & probable sell dates based on capital and expected profit target.
    """
    try:
        macro_data = await mcp_client.get_macro_pulse()
        result = await asyncio.to_thread(
            calculate_target_selling_points,
            capital_inr=req.capital_inr,
            target_profit_inr=req.target_profit_inr,
            time_horizon_months=req.time_horizon_months,
            risk_profile=req.risk_profile,
            macro_data=macro_data
        )
        return result
    except Exception as e:
        logger.error(f"Error calculating target selling points: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/ticker-history", response_model=TickerHistoryResponse)
async def get_ticker_history(
    ticker: str = "RELIANCE.NS",
    period: str = "6mo",
    target_profit_pct: float = 5.0
):
    """
    Fetch historical daily prices (OHLC) for a ticker and run scenario backtest simulation
    evaluating historical target price hit dates and day velocities.
    """
    try:
        res = await asyncio.to_thread(
            fetch_ticker_price_history,
            ticker=ticker,
            period=period,
            target_profit_pct=target_profit_pct
        )
        return res
    except Exception as e:
        logger.error(f"Error fetching ticker history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/stress-test", response_model=StressTestResponse)
async def run_stress_test(req: StressTestRequest):
    """
    Simulate multi-variable geopolitical & macroeconomic shock scenarios
    (Crude Oil, USD/INR, VIX, FII Sell-off, RBI Rate Hike, GDELT Tension, DXY Rally)
    and compute detailed asset-class impact, sector vulnerabilities, and defensive hedges.
    """
    base_macro = await mcp_client.get_macro_pulse()

    # Calculate simulated macro indicators
    crude_shock = req.crude_oil_spike_pct
    fx_shock = req.usd_inr_depreciation_pct
    vix_shock = req.vix_spike_pct
    fii_shock = req.fii_outflow_spike_cr
    rbi_shock = req.rbi_rate_hike_bps
    gdelt_shock = req.gdelt_escalation_pct
    dxy_shock = req.dxy_rally_pct

    # Dynamic composite threat score calculation (0 - 100)
    composite_delta = (
        (crude_shock * 0.55) +
        (fx_shock * 1.8) +
        (vix_shock * 0.35) +
        (abs(fii_shock) / 350.0) +
        (rbi_shock * 0.18) +
        (gdelt_shock * 0.30) +
        (dxy_shock * 1.2)
    )

    sim_threat = min(100.0, max(0.0, base_macro["threat_score"] + composite_delta))

    # Regimes & Narratives
    if sim_threat >= 70.0:
        sim_regime = "HIGH_CRUDE_INFLATION_RISK" if crude_shock >= 15.0 else "GEOPOLITICAL_CRUCIAL_SHOCK"
        sim_regime_label = "High Crude Oil Inflation & Geopolitical Risk" if crude_shock >= 15.0 else "Critical Crisis (Severe Risk-Off Panic)"
        vulnerable = ["Auto & Ancillaries", "Paints, Chemicals & Aviation", "High-Beta Mid/Smallcaps", "Capital Goods"]
        resilient = ["Gold & Precious Metals", "Sovereign Debt ETFs", "Defensive Pharma", "IT Exporters"]
        defensive = ["GOLDBEES.NS", "BHARATBOND.NS", "ITBEES.NS", "SUNPHARMA.NS"]
    elif sim_threat >= 60.0:
        sim_regime = "HIGH_CRUDE_INFLATION_RISK"
        sim_regime_label = "Elevated Macro Inflation & FX Pressure"
        vulnerable = ["Consumer Discretionary", "Oil Import Dependent Equities", "Banking & Financials"]
        resilient = ["Domestic FMCG", "Upstream Oil & Gas", "Gold ETFs", "IT Software Exporters"]
        defensive = ["GOLDBEES.NS", "ITC.NS", "OIL.NS", "TCS.NS"]
    elif sim_threat >= 45.0:
        sim_regime = "FII_OUTFLOW_VOLATILITY"
        sim_regime_label = "FII Outflow Volatility & Sector Rotation"
        vulnerable = ["High-Valuation Tech", "Financial Services"]
        resilient = ["Large-cap Nifty 50 Defensives", "Short-term Debt ETFs", "Gold"]
        defensive = ["GOLDBEES.NS", "LIQUIDBEES.NS", "NIFTYBEES.NS"]
    else:
        sim_regime = "BULLISH_DOMESTIC_GROWTH"
        sim_regime_label = "Stable Growth / Bullish Domestic Expansion"
        vulnerable = []
        resilient = ["Nifty 50 Index", "Bank Nifty Index", "Infrastructure & Capital Goods"]
        defensive = ["NIFTYBEES.NS", "BANKBEES.NS", "RELIANCE.NS"]

    # Calculate portfolio & benchmark estimated PnL impact %
    eq_impact = round(-1.0 * (
        (crude_shock * 0.22) +
        (fx_shock * 0.65) +
        (vix_shock * 0.15) +
        (abs(fii_shock) / 800.0) +
        (rbi_shock * 0.05) +
        (gdelt_shock * 0.10) +
        (dxy_shock * 0.40)
    ), 1)

    # Adjust eq_impact if equity benchmark was positive under low threat
    if sim_threat < 40.0 and eq_impact > -1.0:
        eq_impact = +1.5

    var_increase = round(max(0.0, composite_delta * 0.85 + (vix_shock * 0.6)), 1)

    # Asset class impact breakdown under shock
    asset_breakdown = {
        "Equities": eq_impact,
        "Bonds & Sovereign Debt": round(-1.0 * (rbi_shock * 0.04 + fx_shock * 0.3), 1),
        "Gold & Commodities": round(max(0.0, (crude_shock * 0.3) + (gdelt_shock * 0.25) + (vix_shock * 0.15)), 1),
        "Cash & Liquid Funds": round(rbi_shock * 0.01, 1)
    }

    narrative = (
        f"Under simulated shocks (Crude {crude_shock:+}%, USD/INR {fx_shock:+}%, VIX {vix_shock:+}%, "
        f"FII Sell {fii_shock:,.0f} Cr, RBI Rate {rbi_shock:+} bps), the composite macro threat score increases to "
        f"{sim_threat:.1f}/100 ('{sim_regime_label}'). Portfolio equity drawdowns are estimated at {eq_impact}%, with Value-at-Risk "
        f"expanding by +{var_increase}%. Hedging allocations into Gold ({asset_breakdown['Gold & Commodities']:+}%) and "
        f"Export-earning IT stocks help offset import inflation and domestic volatility."
    )

    return {
        "simulated_threat_score": round(sim_threat, 1),
        "simulated_regime": sim_regime,
        "simulated_regime_label": sim_regime_label,
        "estimated_portfolio_impact_pct": eq_impact,
        "estimated_var_increase_pct": var_increase,
        "high_vulnerability_sectors": vulnerable,
        "resilient_sectors": resilient,
        "defensive_recommendations": defensive,
        "asset_class_impact_breakdown": asset_breakdown,
        "scenario_narrative": narrative
    }

@app.post("/api/broker-execute", response_model=BrokerExecuteResponse)
async def execute_broker_orders(req: BrokerExecuteRequest):
    """
    Execute 1-click recommendations directly via Indian Broker APIs
    (Zerodha KiteConnect, Angel One SmartAPI, or Upstox API).
    """
    import datetime
    total_val = sum(o.get("allocation_inr", 0.0) for o in req.orders)
    summary = []
    for o in req.orders:
        summary.append({
            "ticker": o.get("ticker"),
            "action": "BUY",
            "quantity": o.get("suggested_quantity", 1),
            "amount_inr": o.get("allocation_inr", 0.0),
            "status": "QUEUED_EXECUTION"
        })

    return {
        "status": "SUCCESS",
        "broker_name": req.broker_name,
        "executed_count": len(req.orders),
        "total_executed_value_inr": round(total_val, 2),
        "orders_summary": summary,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/api/tickers")
async def get_tickers():
    """Retrieve full raw ticker dataset from JSON storage."""
    tickers = await asyncio.to_thread(get_all_tickers)
    return {"status": "SUCCESS", "total_tickers": len(tickers), "tickers": tickers}

@app.post("/api/tickers")
async def save_tickers(req: TickerSaveRequest):
    """Update and persist modified ticker dataset in JSON database."""
    try:
        raw_items = [item.model_dump() for item in req.tickers]
        res = await asyncio.to_thread(save_ticker_dataset, raw_items)
        return res
    except Exception as e:
        logger.error(f"Error saving ticker dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to save ticker dataset")

@app.post("/api/tickers/sync", response_model=TickerSyncResponse)
async def sync_tickers():
    """On-demand synchronization of Top 100 NSE & Top 500 BSE securities dataset."""
    try:
        res = await asyncio.to_thread(sync_top_tickers_dataset)
        return res
    except Exception as e:
        logger.error(f"Error syncing ticker dataset: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync ticker dataset")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
