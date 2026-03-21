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
    text = text.lower()
    
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
def classify_with_llm(text):
    prompt = f"""
    You are a financial transaction classifier. Classify this transaction into one of the following categories:

    Income from Mobile Money, Income from Bank, Deposited to M-PESA, Mobile Money Transfer,
    Transaction Cost, Shopping, Airtime Purchase, Internet Bundles Buy, Paybill Payment,
    Withdrawn from MPESA, Online Bill, Utility Bill, Transport

    Transaction:
    {text}

    Return only the category inside <category></category>.
    Do not include any explanation, only the category tag.
    """
    chat_completion = groq.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    content = chat_completion.choices[0].message.content
    match = re.search(r'<category>(.*)<\/category>', content, flags=re.DOTALL)
    category = "Unclassified"
    if match:
        category = match.group(1).strip()
    return category
    
# Main classifier
def classify_transaction(text):
    # Regex first
    label = classify_with_regex(text)
    if label:
        return label, "regex"

    # Fallback to LLM
    llm_label = classify_with_llm(text)
    return llm_label, "llm"

# CSV classification
def classify_csv(input_file, output_file="mpesa_classified.csv"):
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_file = BASE_DIR / "resources" / input_file
    output_file = BASE_DIR / "resources" / output_file

    if not input_file.exists():
        raise FileNotFoundError(f"{input_file} not found.")

    df = pd.read_csv(input_file)

    # Classify
    results = df["Details"].apply(classify_transaction)

    df["Predicted_Label"] = results.apply(lambda x: x[0])
    df["Method"] = results.apply(lambda x: x[1])

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