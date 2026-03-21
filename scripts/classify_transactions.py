import pandas as pd
import re
from dotenv import load_dotenv
from groq import Groq
import os
from pathlib import Path

load_dotenv()
groq = Groq()

# Regex classifier
def classify_with_regex(text):
    if pd.isna(text):
        text = ""
    text = str(text).lower()
    
    patterns = {
        r"funds received from": "Income from Mobile Money",
        r"business payment from": "Income from Bank",
        r"deposit of funds at agent till": "Deposited to M-PESA",
        r"customer transfer to": "Mobile Money Transfer",
        r"customer transfer of funds charge": "Transaction Cost",
        r"merchant payment": "Shopping",
        r"airtel money": "Mobile Money Transfer",
        r"offnet c2b transfer": "Mobile Money Transfer",
        r"kplc|water": "Utility Bill",
        r"airtime purchase": "Airtime Purchase",
        r"bundle purchase": "Internet Bundles Buy",
        r"pay bill charge": "Transaction Cost",
        r"pay bill": "Paybill Payment",
        r"withdrawal charge": "Transaction Cost",
        r"customer withdrawal at agent till": "Withdrawn from MPESA",
        r"customer payment to small business": "Shopping",
        r"pay bill online": "Online Bill",
        r"supermarket|naivas|carrefour|shop|store|mart": "Shopping",
        r"uber|bolt|taxi": "Transport",
    }
    
    for pattern, label in patterns.items():
        if re.search(pattern, text):
            return label
    
    return None

# LLM fallback
def classify_transactions_batch(text_list):
    """
    text_list: list of transaction strings
    returns: list of categories in the same order
    """
    # Build prompt for all transactions at once
    prompt = "You are a financial transaction classifier. Classify each transaction into one of the following categories:\n\n"
    prompt += "Income from Mobile Money, Income from Bank, Deposited to M-PESA, Mobile Money Transfer, Transaction Cost, Shopping, Airtime Purchase, Internet Bundles Buy, Paybill Payment, Withdrawn from MPESA, Online Bill, Utility Bill, Transport\n\n"
    prompt += "Return the category for each transaction on a separate line using <category></category> tags. Do NOT include explanations.\n\n"
    prompt += "Transactions:\n"
    for i, text in enumerate(text_list, 1):
        prompt += f"{i}. {text}\n"

    chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )

    content = chat_completion.choices[0].message.content
    categories = []

    # Extract all <category> tags line by line
    for match in re.findall(r'<category>(.*?)<\/category>', content, flags=re.DOTALL):
        categories.append(match.strip())

    # If the number of categories returned < number of transactions, pad with "Unclassified"
    while len(categories) < len(text_list):
        categories.append("Unclassified")

    return categories

# CSV classification
def classify_csv(input_file, output_file="mpesa_classified.csv"):
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_file = BASE_DIR / "resources" / input_file
    output_file = BASE_DIR / "resources" / output_file

    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found.")

    df = pd.read_csv(input_file)

    # Classify transactions using batch LLM + regex fallback
    llm_needed_mask = df["Details"].apply(lambda x: classify_with_regex(x) is None)
    llm_texts = df.loc[llm_needed_mask, "Details"].tolist()
    llm_categories = classify_transactions_batch(llm_texts)
    df.loc[llm_needed_mask, "Subcategory"] = llm_categories
    df.loc[llm_needed_mask, "Method"] = "llm"
    
    # Assigning regex classifications for the rest
    df.loc[~llm_needed_mask, "Subcategory"] = df.loc[~llm_needed_mask, "Details"].apply(classify_with_regex)
    df.loc[~llm_needed_mask, "Method"] = "regex"

    # Save output
    df.to_csv(output_file, index=False)
    print(f"Classification complete. Output saved to '{output_file}'.")

    return output_file

# Script execution
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    input_file = BASE_DIR / "resources" / "clean_mpesa_transactions.csv"

    # Run classification
    classify_csv(input_file.name) 
