import streamlit as st
import pandas as pd
import pikepdf
import pdfplumber
import re
from pathlib import Path
from io import BytesIO
import plotly.express as px

from dotenv import load_dotenv
from groq import Groq
import os

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

# Main transaction classifier
def classify_transaction(text):
    label = classify_with_regex(text)
    if label:
        return label, "regex"
    return classify_with_llm(text), "llm"


# Clean details
def clean_details(text):
    if pd.isna(text):
        return ""
    return " ".join(str(text).split())


# Extract transaction notes
def extract_notes(text):
    if pd.isna(text):
        return None
    match = re.search(r"-\s*(.+)", text)
    if not match:
        return None
    segment = match.group(1)
    segment = re.split(r"\.| for | Acc\.|Original|by|via", segment)[0]
    segment = re.sub(r"\b\d[\d\*]*\b", "", segment)
    segment = " ".join(segment.split())
    return segment.strip() if segment else None


# Clean numeric columns
def extract_amount(value):
    if pd.isna(value):
        return 0.0
    value = str(value).replace("-", "").strip()
    try:
        return float(value)
    except ValueError:
        return 0.0

def label_transaction(amount):
    if amount > 0:
        return "Income"
    elif amount < 0:
        return "Expense"
    else:
        return "Neutral"


# Streamlit UI
st.set_page_config(page_title="M-PESA Transactions Classifier", layout="wide")
st.title("M-PESA Statement Cleaner & Classifier")

uploaded_file = st.file_uploader("Upload your M-PESA PDF statement", type=["pdf"])

if uploaded_file:
    # Require password
    pdf_password = st.text_input("Enter PDF password (required)", type="password")
    if not pdf_password:
        st.warning("Kindly enter the PDF password to unlock.")
        st.stop()

    # Save to BytesIO for processing
    pdf_bytes = BytesIO(uploaded_file.read())
    unlocked_bytes = BytesIO()

    try:
        with pikepdf.open(pdf_bytes, password=pdf_password) as pdf:
            pdf.save(unlocked_bytes)
        st.success("PDF unlocked successfully!")
        st.info("Running classification. Please wait.....")
        
    except Exception as e:
        st.error(f"Error unlocking PDF: {e}")
        st.stop()

    # Extract tables
    all_rows = []
    with pdfplumber.open(unlocked_bytes) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                all_rows.extend(table)

    if not all_rows:
        st.warning("No tables found in PDF.")
        st.stop()

    df = pd.DataFrame(all_rows)

    # Set column names
    df.columns = [
        "Receipt No",
        "Completion Time",
        "Details",
        "Transaction Status",
        "Paid In",
        "Withdrawn",
        "Balance"
    ]

    # Remove repeated headers
    header_values = df.columns.tolist()
    df = df[~df.apply(lambda row: all(any(h.lower() in str(cell).lower() for h in header_values) for cell in row), axis=1)].reset_index(drop=True)

    # Clean details & extract notes
    df["Details"] = df["Details"].apply(clean_details)
    df["Notes"] = df["Details"].apply(extract_notes)

    # Processing the time column
    df['Completion Time'] = df['Completion Time'].astype(str).str.strip()
    df['Completion Time'] = pd.to_datetime(df['Completion Time'], errors='coerce', infer_datetime_format=True)
    
    df['Year'] = df['Completion Time'].dt.year
    df['Month'] = df['Completion Time'].dt.strftime('%b')
    df['Date'] = df['Completion Time'].dt.day
    df['Day'] = df['Completion Time'].dt.day_name()

    # Clean numeric columns
    for col in ["Paid In", "Withdrawn", "Balance"]:
        df[col] = df[col].apply(extract_amount)
    df["Amount"] = df["Paid In"] - df["Withdrawn"]
    df["Category"] = df["Amount"].apply(label_transaction)

    # Classify transactions
    results = df["Details"].apply(classify_transaction)
    df["Subcategory"] = results.apply(lambda x: x[0])
    df["Method"] = results.apply(lambda x: x[1])

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Classified CSV",
        data=csv_bytes,
        file_name="mpesa_classified.csv",
        mime="text/csv"
    )


    
    
    # -------------------------------
    # Streamlit Visualization Section
    # -------------------------------
    st.subheader("Transaction Summary Visualizations")
    
    # Filtering out Neutral transactions and taking absolute amounts
    df_viz = df[df["Category"] != "Neutral"].copy()
    df_viz["Abs_Amount"] = df_viz["Amount"].abs()
    
    col1, col2 = st.columns([1, 2])
    
    # -------------------------------
    # Left Column: Total Income vs Expenses
    # -------------------------------
    with col1:
        st.markdown("### Income vs Expenses")
    
        # Time filter
        time_filter = st.radio(
            "Filter transactions by:", ["All Time", "Year", "Month", "Week", "Day of Week"]
        )
    
        filtered_df = df_viz.copy()
    
        if time_filter == "Year":
            year_options = sorted(filtered_df['Year'].dropna().unique())
            selected_year = st.selectbox("Select Year:", year_options)
            filtered_df = filtered_df[filtered_df['Year'] == selected_year]
    
        elif time_filter == "Month":
            month_options = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            selected_month = st.selectbox("Select Month:", month_options)
            filtered_df = filtered_df[filtered_df['Month'] == selected_month]
    
        elif time_filter == "Week":
            if 'Week' not in filtered_df.columns:
                filtered_df['Week'] = filtered_df['Completion Time'].dt.isocalendar().week
            week_options = sorted(filtered_df['Week'].dropna().unique())
            selected_week = st.selectbox("Select Week Number:", week_options)
            filtered_df = filtered_df[filtered_df['Week'] == selected_week]
    
        elif time_filter == "Day of Week":
            day_options = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            selected_day = st.selectbox("Select Day:", day_options)
            filtered_df = filtered_df[filtered_df['Day'] == selected_day]
    
        # Total amounts by Category
        category_summary = filtered_df.groupby("Category")["Abs_Amount"].sum().reset_index()
    
        if not category_summary.empty:
            fig_total = px.bar(
                category_summary,
                x="Category",
                y="Abs_Amount",
                color="Category",
                text="Abs_Amount",
                labels={"Abs_Amount":"Amount (Ksh)"},
                title="Income vs Expenses"
            )
            fig_total.update_layout(showlegend=False)
            st.plotly_chart(fig_total, use_container_width=True)
        else:
            st.info("No transactions for the selected time filter.")
    
    # -------------------------------
    # Right Column: Subcategory Breakdown
    # -------------------------------
    with col2:
        st.markdown("### Subcategory Breakdown")
    
        if not filtered_df.empty:
            # Category selection for breakdown
            category_options = filtered_df["Category"].unique().tolist()
            selected_cat = st.selectbox("Select Category to visualize:", category_options)
            cat_df = filtered_df[filtered_df["Category"] == selected_cat].copy()
    
            # Subcategory chart
            subcat_summary = (
                cat_df.groupby("Subcategory")["Abs_Amount"]
                .sum()
                .reset_index()
                .sort_values("Abs_Amount", ascending=True)
            )
    
            fig_subcat = px.bar(
                subcat_summary,
                x="Abs_Amount",
                y="Subcategory",
                orientation="h",
                color="Subcategory",
                text="Abs_Amount",
                labels={"Abs_Amount":"Amount (Ksh)", "Subcategory":"Subcategory"},
                title=f"{selected_cat} Breakdown"
            )
            fig_subcat.update_layout(showlegend=False)
            st.plotly_chart(fig_subcat, use_container_width=True)
    
            # Subcategory Notes / Transaction Details
            subcat_options = cat_df["Subcategory"].unique().tolist()
            selected_subcat = st.selectbox("Select Subcategory to see transaction details:", subcat_options)
    
            notes_df = (
                cat_df[cat_df["Subcategory"] == selected_subcat]
                .groupby("Notes")["Abs_Amount"]
                .sum()
                .reset_index()
                .sort_values("Abs_Amount", ascending=True)
            )
    
            if not notes_df.empty:
                fig_notes = px.bar(
                    notes_df,
                    x="Abs_Amount",
                    y="Notes",
                    orientation="h",
                    color="Notes",
                    text="Abs_Amount",
                    labels={"Abs_Amount":"Amount (Ksh)", "Notes":"Transaction Details"},
                    title=f"{selected_subcat} Transaction Details"
                )
                fig_notes.update_layout(showlegend=False)
                st.plotly_chart(fig_notes, use_container_width=True)
            else:
                st.info("No transaction details available for this Subcategory.")
        else:
            st.info("No transactions for the selected time filter.")
