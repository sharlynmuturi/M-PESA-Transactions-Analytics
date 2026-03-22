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
        return None
        
    text = text.lower()

    patterns = [
        # Transaction costs (first to avoid misclassification)
        (r"customer transfer of funds charge", "Transaction Costs (Send Money)"),
        (r"pay merchant charge", "Transaction Costs (Pochi la Biashara)"),
        (r"pay bill charge", "Transaction Costs (Paybill)"),
        (r"withdrawal charge", "Transaction Costs (Withdraw)"),

        # Deposits / withdrawals
        (r"deposit of funds at agent till", "M-PESA Deposits"),
        (r"m-shwari deposit", "M-Shwari Deposits"),
        (r"customer withdrawal at agent till", "MPESA Withdrawals"),
        (r"m-shwari withdraw", "M-Shwari Withdrawals"),

        # Incoming
        (r"funds received from", "Received (Send Money)"),
        (r"business payment from", "Received (Bank)"),
        (r"offnet b2c transfer", "Received (Send Money)"),

        # Outgoing
        (r"offnet c2b transfer", "Sent"),
        (r"customer transfer to", "Sent"),

        # Shopping
        (r"merchant payment", "Shopping (Till)"),
        (r"customer payment to small business", "Shopping (Pochi la Biashara)"),

        # Bills
        (r"pay bill online", "Bills (Online)"),
        (r"pay bill to", "Bills (Paybill)"),
        (r"kplc prepaid", "Bills (Electricity)"),
        (r"bundle purchase|customer bundle purchase|safaricom data bundles", "Bills (Data Bundles)"),
        (r"airtime purchase", "Bills (Airtime Purchase)"),

        # Reversals
        (r"reversal", "Reversals"),
    ]

    for pattern, label in patterns:
        if re.search(pattern, text):
            return label

    return None

# LLM fallback
def classify_transactions_batch(text_list):
    """
    text_list: list of transaction strings
    Returns: list of categories in the same order
    """
    # Build prompt for all transactions at once
    prompt = "You are a financial transaction classifier. Classify each transaction into one of the following categories:\n\n"
    prompt += "M-PESA Deposits, M-Shwari Deposits, MPESA Withdrawals, M-Shwari Withdrawals, Received (Send Money), Received (Bank), Sent (Send Money), Shopping (Till), Shopping (Pochi la Biashara),Bills (Online), Bills (Paybill), Bills (Electricity), Bills (Data Bundles), Bills (Airtime Purchase), Transaction Costs (Send Money), Transaction Costs (Paybill), Transaction Costs (Withdraw), Reversals, Unclassified.\n\n"
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

    valid_categories = {
        "M-PESA Deposits",
        "M-Shwari Deposits",
        "MPESA Withdrawals",
        "Received (Send Money)",
        "Received (Bank)",
        "Sent",
        "Shopping (Till)",
        "Shopping (Pochi la Biashara)",
        "Bills (Online)",
        "Bills (Paybill)",
        "Bills (Electricity)",
        "Bills (Data Bundles)",
        "Bills (Airtime Purchase)",
        "Transaction Costs (Send Money)",
        "Transaction Costs (Paybill)",
        "Transaction Costs (Withdraw)",
        "Reversals",
        "Unclassified"
    }

    # Extraction
    categories = []

    for match in re.findall(r'<category>(.*?)<\/category>', content, flags=re.DOTALL):
        cat = match.strip()
        if cat not in valid_categories:
            cat = "Unclassified"
        categories.append(cat)
        
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
    llm_indices = df.index[llm_needed_mask].tolist()
    
    llm_texts = df.loc[llm_needed_mask, "Details"].tolist()
    
    if llm_texts:
        llm_categories = classify_transactions_batch(llm_texts)
    
        # Ensure exact length match
        if len(llm_categories) != len(llm_indices):
            llm_categories = llm_categories[:len(llm_indices)]
            llm_categories += ["Unclassified"] * (len(llm_indices) - len(llm_categories))
    
        # Assign row by row
        for idx, cat in zip(llm_indices, llm_categories):
            df.at[idx, "Subcategory"] = cat
            df.at[idx, "Method"] = "llm"
    
    # Assigning regex classifications for the rest
    df.loc[~llm_needed_mask, "Subcategory"] = df.loc[~llm_needed_mask, "Details"].apply(classify_with_regex)
    df.loc[~llm_needed_mask, "Method"] = "regex"

    # Split subcategory into parent + child
    def split_subcategory(subcat):
        if pd.isna(subcat):
            return ("Unclassified", "Unclassified")
        
        match = re.match(r"(.+?)\s*\((.+)\)", subcat)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        else:
            return subcat.strip(), subcat.strip()
    
    df[["MainCategory", "SubCategoryDetail"]] = df["Subcategory"].apply(
        lambda x: pd.Series(split_subcategory(x))
    )
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
