# Crude Oil Inventory Event Strategy (WTI Term-Structure Proxy)

## Overview
This project develops an event-driven quantitative trading framework to analyze how WTI crude oil term structure responds to weekly U.S. Energy Information Administration (EIA) inventory releases.

The study investigates whether standardized inventory surprises contain predictive information for short-term spread movements and evaluates this through a systematic backtesting pipeline.

---

## Key Features
- Automated ingestion of EIA crude inventory data (2010–2026)
- Construction of WTI term-structure proxy from market data
- Event-driven signal generation using rolling z-score normalization
- Backtesting framework with transaction costs and fixed holding windows
- Event study analysis to quantify post-release spread dynamics
- Full visualization suite (PnL, signals, positions, event response)

---

## Strategy Logic
1. Compute expected inventory using rolling average  
2. Define surprise = actual − expected  
3. Standardize using rolling z-score  
4. Generate signals:
   - Long spread if z ≤ -1  
   - Short spread if z ≥ 1  
5. Hold positions for a fixed 5-day window  

---

## Results Summary

### Strategy Performance
- Total Return: **-13.79**
- Annualized Return: **-1.23**
- Sharpe Ratio: **-0.03**
- Max Drawdown: **-110.45**
- Trade Win Rate: **51.28%**

### Event Study Insights
- Avg Spread Change (t+1): **+0.099**
- Avg Spread Change (t+5): **+1.364**
- Positive Response Rate: **~65%**
- Peak Response: **t+5**

---

## Key Insight
The results show a strong divergence between signal quality and trading performance:

> Inventory surprises exhibit statistically meaningful predictive power, but naive execution fails to capture this alpha effectively.

This suggests that the limitation lies in strategy design rather than signal strength.

---

## Limitations
- Spread constructed as a proxy rather than true CL1–CL2 futures spread  
- No slippage or liquidity modeling  
- Fixed holding period may not align with signal decay  
- No regime or volatility filtering  

---

## Future Improvements
- Use actual front-month vs next-month futures spread  
- Optimize holding periods and thresholds  
- Incorporate volatility targeting  
- Apply machine learning for signal refinement  

---

## Tech Stack
- Python (pandas, numpy, matplotlib)
- EIA API for macroeconomic data
- Time series analysis and backtesting framework

---
