import pikepdf
import pdfplumber
import pandas as pd
import re

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
            
pdf_input = BASE_DIR / "resources" / "mpesa_statement.pdf"
pdf_password = input("Enter the File password: ").strip()
pdf_unlocked = BASE_DIR / "resources" / "unlocked_statement.pdf"

print(f"Unlocking PDF '{pdf_input}'...")
with pikepdf.open(pdf_input, password=pdf_password if pdf_password else None) as pdf:
    pdf.save(pdf_unlocked)
print(f"Saved unlocked PDF as '{pdf_unlocked}'")

# Extracting tables from PDF
all_rows = []

print("Extracting tables from PDF...")
with pdfplumber.open(pdf_unlocked) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        table = page.extract_table()
        if table:
            all_rows.extend(table)

df = pd.DataFrame(all_rows)
print(f"Extracted {len(df)} rows from PDF.")


# Setting column names
df.columns = [
    "Receipt No",
    "Completion Time",
    "Details",
    "Transaction Status",
    "Paid In",
    "Withdrawn",
    "Balance"
]


# Remove repeated header rows
header_values = ["Receipt No", "Completion Time", "Details", "Transaction Status", "Paid In", "Withdrawn", "Balance"]

def is_header_row(row):
    """Check if every cell contains at least part of a header"""
    return all(any(h.lower() in str(cell).lower() for h in header_values) for cell in row)

df = df[~df.apply(is_header_row, axis=1)].reset_index(drop=True)
print("Removed repeated header rows.")

# Clean Details
def clean_details(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())

df["Details"] = df["Details"].apply(clean_details)

# Key description notes
def extract_notes(text):
    if pd.isna(text):
        return None
    
    text = str(text)
    
    match = re.search(r"-\s*(.+)", text)
    if not match:
        return None
    
    segment = match.group(1)
    segment = re.split(r"\.| for | Acc\.|Original|by|via", segment)[0]
    segment = re.sub(r"\b\d[\d\*]*\b", "", segment)
    segment = " ".join(segment.split())
    
    return segment.strip() if segment else None

df["Notes"] = df["Details"].apply(extract_notes)

# Processing the time column
df['Completion Time'] = df['Completion Time'].astype(str).str.strip()
df['Completion Time'] = pd.to_datetime(df['Completion Time'], errors='coerce')

df['Year'] = df['Completion Time'].dt.year
df['Month'] = df['Completion Time'].dt.strftime('%b')
df['Date'] = df['Completion Time'].dt.day
df['Day'] = df['Completion Time'].dt.day_name()

# Clean numeric columns
def extract_amount(value):
    if pd.isna(value):
        return 0.0
    value = str(value)
    value = value.replace("-", "").strip()
    try:
        return float(value)
    except ValueError:
        return 0.0

for col in ["Paid In", "Withdrawn", "Balance"]:
    df[col] = df[col].apply(extract_amount)

# Compute amount
df["Amount"] = df["Paid In"] - df["Withdrawn"]

# Label income or expense or neutral
def label_transaction(amount):
    if amount > 0:
        return "Income"
    elif amount < 0:
        return "Expense"
    else:
        return "Neutral"

df["Category"] = df["Amount"].apply(label_transaction)


# Save cleaned CSV
output_csv = BASE_DIR / "resources" / "clean_mpesa_transactions.csv"
df.to_csv(output_csv, index=False)
print(f"Cleaned dataset saved to '{output_csv}'")
print("Done!")