import os
import yfinance as yf
from openai import OpenAI

# Initialize OpenAI client with environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def safe_round(value, decimals=2):
    try:
        if value is None or value == "N/A":
            return value
        return round(float(value), decimals)
    except Exception:
        return value

def stock_analyzer(symbol: str):
    """
    Fetch stock data and return a structured analysis and LLM verdict.
    """

    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        # Basic company summary
        company_summary = info.get("longBusinessSummary", "No summary available.")

        # Calculate Free Cashflow Yield
        try:
            fcf = stock.cashflow.loc["Total Cash From Operating Activities"].iloc[0] \
                  - stock.cashflow.loc["Capital Expenditures"].iloc[0]
            market_cap = info.get("marketCap", 1)
            fcf_yield = fcf / market_cap if market_cap else None
        except Exception:
            fcf_yield = None

        # Calculate Return on Capital Employed (ROCE)
        try:
            ebit = stock.financials.loc["Ebit"].iloc[0]
            total_assets = stock.balance_sheet.loc["Total Assets"].iloc[0]
            current_liabilities = stock.balance_sheet.loc["Total Current Liabilities"].iloc[0]
            roce = ebit / (total_assets - current_liabilities) if ebit and total_assets and current_liabilities else None
        except Exception:
            roce = None

        # Calculate Revenue Growth Rate
        try:
            revenues = stock.financials.loc["Total Revenue"]
            if len(revenues) > 1:
                revenue_growth = (revenues.iloc[0] - revenues.iloc[1]) / revenues.iloc[1]
            else:
                revenue_growth = None
        except Exception:
            revenue_growth = None

        # Calculate EPS Growth Rate
        try:
            eps_current = info.get("trailingEps")
            eps_forward = info.get("forwardEps")
            eps_growth = (eps_forward - eps_current) / eps_current if eps_current and eps_forward else None
        except Exception:
            eps_growth = None


        # Key financials
        financials = {
            # Column 1
            "Price-to-Earnings (P/E)": safe_round(info.get("trailingPE", "N/A")),
            "Price-to-Book (P/B)": safe_round(info.get("priceToBook", "N/A")),
            "Price-to-Sales (P/S)": safe_round(info.get("priceToSalesTrailing12Months", "N/A")),
            "Price-to-Earnings-Growth (PEG)": safe_round(info.get("pegRatio", "N/A")),
            "Free Cash Flow (FCF) Yield": safe_round(fcf_yield),
            "Earnings Per Share (EPS)": safe_round(info.get("trailingEps", "N/A")),
            "Return on Equity (ROE)": safe_round(info.get("returnOnEquity", "N/A")),
            "Return on Assets (ROA)": safe_round(info.get("returnOnAssets", "N/A")),
            "Return on Capital Employed (ROCE)": safe_round(roce),

            # Column 2
            "Gross Profit Margin": safe_round(info.get("grossMargins", "N/A")),
            "Operating Margin": safe_round(info.get("operatingMargins", "N/A")),
            "Net Profit Margin": safe_round(info.get("profitMargins", "N/A")),
            "Debt-to-Equity (D/E)": safe_round(info.get("debtToEquity", "N/A")),
            "Revenue Growth Rate": safe_round(revenue_growth),
            "EPS Growth Rate": safe_round(eps_growth),
            "Dividend Yield": safe_round(info.get("dividendYield", "N/A")),
            "Current Ratio": safe_round(info.get("currentRatio", "N/A")),
            "Quick Ratio": safe_round(info.get("quickRatio", "N/A")),
        }

        # Analyst data (live data from yfinance)
        try:
            recs = stock.recommendations_summary
            if recs is not None and not recs.empty:
                latest = recs.iloc[-1]
                analyst_ratings = {
                    "Strong Buy": recs.get("Strong Buy", 0),
                    "Buy": recs.get("Buy", 0),
                    "Hold": recs.get("Hold", 0),
                    "Underperform": recs.get("Underperform", 0),
                    "Sell": recs.get("Sell", 0),
                }
            else:
                analyst_ratings = {"buy": 0, "hold": 0, "sell": 0}
        except Exception:
            analyst_ratings = {"buy": 0, "hold": 0, "sell": 0}

        # Placeholder AI verdict (later we’ll add Hugging Face/OpenAI)
        verdict = run_llm_analysis(financials, company_summary)

        return {
            "symbol": symbol.upper(),
            "company_summary": company_summary,
            "financials": financials,
            "analyst_ratings": analyst_ratings,
            "verdict": verdict,
        }

    except Exception as e:
        return {
            "error": f"Could not fetch data for {symbol}: {str(e)}"
        }
    
def run_llm_analysis(financials, company_summary):
    """
    Use OpenAI to generate a stock analysis verdict.
    """
    # Key metrics used to feed into the LLM
    key_metrics = {
        "P/E": financials.get("Price-to-Earnings (P/E)"),
        "P/B": financials.get("Price-to-Book (P/B)"),
        "P/S": financials.get("Price-to-Sales (P/S)"),
        "PEG": financials.get("Price-to-Earnings-Growth (PEG)"),
        "FCF Yield": financials.get("Free Cash Flow (FCF) Yield"),
        "ROE": financials.get("Return on Equity (ROE)"),
        "ROA": financials.get("Return on Assets (ROA)"),
        "ROCE": financials.get("Return on Capital Employed (ROCE)"),
        "Revenue Growth": financials.get("Revenue Growth Rate"),
        "EPS Growth": financials.get("EPS Growth Rate"),
        "Dividend Yield": financials.get("Dividend Yield"),
        "Debt/Equity": financials.get("Debt-to-Equity (D/E)"),
    }
    
    prompt = f"""
    You are a professional equity analyst. Analyze the following stock:

Company Summary: {company_summary[:500]}...
Financials: {key_metrics}

Please provide a concise analysis (3–5 sentences) that covers:
1. Valuation (undervalued, overvalued, fair).
2. Profitability and growth outlook.
3. Whether this stock represents a good buy opportunity.

Keep your tone professional and data-driven.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert equity analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()