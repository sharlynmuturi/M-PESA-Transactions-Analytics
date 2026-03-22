import streamlit as st
import pandas as pd
import pikepdf
import pdfplumber
import re
from pathlib import Path
from io import BytesIO
import altair as alt
import time

from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

groq_api_key = st.secrets["GROQ_API_KEY"]
groq = Groq(api_key=groq_api_key)

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
@st.cache_data(show_spinner=False)
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
        "Sent (Send Money)",
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
st.title("M-PESA Transactions Analytics Platform")

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
        st.success("PDF unlocked successfully! Analyzing transactions...")
        
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
    df['Week'] = df['Completion Time'].dt.isocalendar().week

    # Clean numeric columns
    for col in ["Paid In", "Withdrawn", "Balance"]:
        df[col] = df[col].apply(extract_amount)
    df["Amount"] = df["Paid In"] - df["Withdrawn"]
    df["Category"] = df["Amount"].apply(label_transaction)

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

    # Download CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Classified CSV",
        data=csv_bytes,
        file_name="mpesa_classified.csv",
        mime="text/csv"
    )
    
   # Visualizations
    st.header("Transaction Visualizations")

    # Filtering out Neutral
    df_viz = df[df["Category"] != "Neutral"].copy()

    # Taking absolute values for expenses
    df_viz["Abs_Amount"] = df_viz["Amount"].abs()


    # Time filter
    time_filter = st.radio("Filter transactions by:", ["All Time", "Year", "Month", "Week", "Day of Week"])

    col1, col2 = st.columns([1, 2])

    # Left

    with col1:
        
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
            week_options = sorted(filtered_df['Week'].dropna().unique())
            selected_week = st.selectbox("Select Week Number:", week_options)
            filtered_df = filtered_df[filtered_df['Week'] == selected_week]
        
        elif time_filter == "Day of Week":
            day_options = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            selected_day = st.selectbox("Select Day:", day_options)
            filtered_df = filtered_df[filtered_df['Day'] == selected_day]

        # Total amounts by category
        st.markdown("### Income vs Expenses")

        category_summary = filtered_df.groupby("Category")["Abs_Amount"].sum().reset_index()
        
        if not category_summary.empty:
            chart_total = alt.Chart(category_summary).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("Category:N", sort=None, title="Category"),


                y=alt.Y("Abs_Amount:Q", title="Amount (Ksh)"),
                color=alt.Color("Category:N", legend=None),
                tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Abs_Amount:Q", format=",.2f")]
            ).properties(width=200, height=400)

            st.altair_chart(chart_total, use_container_width=True)
        else:
            st.info("No transactions for the selected time filter.")
        
        # Net cash Flow trend
        cashflow_df = filtered_df.groupby("Completion Time")["Amount"].sum().reset_index()
        
        chart_cashflow = alt.Chart(cashflow_df).mark_line(point=True).encode(
            x=alt.X("Completion Time:T", title="Date"),
            y=alt.Y("Amount:Q", title="Net Cash Flow (Ksh)"),
            tooltip=["Completion Time:T", "Amount:Q"]
        ).properties(height=200)
        
        st.markdown("### Net Cash Flow Trend")
        st.altair_chart(chart_cashflow, use_container_width=True)
        
    # Right

    with col2:
        st.markdown("### Subcategory Breakdown")
        
        if not filtered_df.empty:
            # select a category to visualize
            category_options = filtered_df["Category"].unique().tolist()
            selected_cat = st.selectbox("Select Category to visualize:", category_options)
            
            cat_df = filtered_df[filtered_df["Category"] == selected_cat].copy()
            
            # Aggregate by main category (before brackets)
            main_summary = (
                cat_df.groupby("MainCategory")["Abs_Amount"]
                .sum()
                .reset_index()
                .sort_values("Abs_Amount", ascending=True)
            )
            
            chart_main = alt.Chart(main_summary).mark_bar(
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3
            ).encode(
                y=alt.Y("MainCategory:N", sort="-x", title="Main Category"),
                x=alt.X("Abs_Amount:Q", title="Amount (Ksh)"),
                color=alt.Color("MainCategory:N", legend=None),
                tooltip=[alt.Tooltip("MainCategory:N"), alt.Tooltip("Abs_Amount:Q", format=",.2f")]
            ).properties(width=500, height=400)
            
            st.altair_chart(chart_main, use_container_width=True)

            # Select MAIN category (e.g. Shopping)
            main_options = cat_df["MainCategory"].unique().tolist()
            selected_main = st.selectbox("Select Subcategory:", main_options)
            
            main_df = cat_df[cat_df["MainCategory"] == selected_main]
            
            # Breakdown into sub-details (Till, Pochi etc.)
            detail_summary = (
                main_df.groupby("SubCategoryDetail")["Abs_Amount"]
                .sum()
                .reset_index()
                .sort_values("Abs_Amount", ascending=True)
            )
            
            chart_detail = alt.Chart(detail_summary).mark_bar(
                cornerRadiusTopLeft=3, cornerRadiusTopRight=3
            ).encode(
                y=alt.Y("SubCategoryDetail:N", sort="-x", title="Subcategory Detail"),
                x=alt.X("Abs_Amount:Q", title="Amount (Ksh)"),
                color=alt.Color("SubCategoryDetail:N", legend=None),
                tooltip=[alt.Tooltip("SubCategoryDetail:N"), alt.Tooltip("Abs_Amount:Q", format=",.2f")]
            ).properties(width=500, height=400)
            
            st.altair_chart(chart_detail, use_container_width=True)            



            # Details breakdown
            detail_options = main_df["SubCategoryDetail"].unique().tolist()
            selected_detail = st.selectbox(
                "See top transaction details in the subcategory:", 
                detail_options
            )
            
            notes_df = (
                main_df[main_df["SubCategoryDetail"] == selected_detail]
                .groupby("Notes")["Abs_Amount"]
                .sum()
                .reset_index()
                .sort_values("Abs_Amount", ascending=False)
                .head(10)
            )

            if not notes_df.empty:
                chart_notes = alt.Chart(notes_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                    y=alt.Y("Notes:N", sort="-x", title="Transaction Details"),
                    x=alt.X("Abs_Amount:Q", title="Amount (Ksh)"),
                    color=alt.Color("Notes:N", legend=None),
                    tooltip=[alt.Tooltip("Notes:N"), alt.Tooltip("Abs_Amount:Q", format=",.2f")]
                ).properties(width=500, height=400)

                st.altair_chart(chart_notes, use_container_width=True)
            else:
                st.info("No detail available for this Subcategory.")
        else:
            st.info("No transactions for the selected time filter.")
