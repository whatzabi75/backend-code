import os
import yfinance as yf
from openai import OpenAI

# Initialize OpenAI client with environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def stock_analyzer(symbol: str):
    """
    Fetch stock data and return a structured analysis and LLM verdict.
    """

    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        # Basic company summary
        company_summary = info.get("longBusinessSummary", "No summary available.")

        # Key financials
        financials = {
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "EPS (TTM)": info.get("trailingEps", "N/A"),
            "Debt/Equity": info.get("debtToEquity", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
        }

        # Placeholder analyst ratings (yfinance doesn’t give full breakdown)
        analyst_ratings = {
            "buy": 12,
            "hold": 4,
            "sell": 2,
        }

        # Placeholder AI verdict (later we’ll add Hugging Face/OpenAI)
        verdict = run_llm_analysis(financials, analyst_ratings, company_summary)

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
    
def run_llm_analysis(financials, analyst_ratings, company_summary):
    """
    Use OpenAI to generate a stock analysis verdict.
    """
    prompt = f"""
    You are a professional equity analyst. Analyze the following stock:

Company Summary: {company_summary[:500]}...
Financials: {financials}
Analyst Ratings: {analyst_ratings}

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