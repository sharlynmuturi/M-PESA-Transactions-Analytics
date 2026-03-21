# M-PESA Transactions Analytics Platform

M-PESA statements are typically exported as PDF documents, often containing **multi-page tables with complex layouts** and messy transaction descriptions. Manually analyzing these statements to understand **income, expenses or spending patterns** is tedious, error-prone, and time-consuming.

This projects uses a **Streamlit web app** to handle the entire pipeline from PDF ingestion to analysis.

**Workflow:**

1.  Request and download a **full M-PESA statement** for your desired time period through the official M-PESA service channels.
2.  Upload the password-protected statement and enter the code sent by Safaricom as **PDF password** to enable unlocking.
2.  **Tables are extracted** from the PDF and **Data cleaning, Classification and Visualization** performed.
3.  **Classification pipeline:**
    *   **Regex-based rules** for quick and accurate labeling
    *   **Fallback LLM (Groq API with LLaMA)** for unrecognized transactions
4.  **Cleaned and classified CSV** is available for download.


* * *

## Technologies Used

## 

| Category | Tool/Library | Purpose |
| --- | --- | --- |
| PDF Handling | pikepdf | Unlock password-protected PDFs |
| PDF Extraction | pdfplumber | Extract tables from PDF pages |
| Data Manipulation | pandas | Clean, transform, aggregate data |
| Regex | re | Extract transaction partners & classify patterns |
| AI/LLM | Groq| LLM fallback for classification |


* * *

## Run Locally

## 

1.  **Clone the repository**

```bash
 git clone https://github.com/sharlynmuturi/M-PESA-Transactions-Narrator.git
cd M-PESA-Transactions-Narrator
```

2.  **Create a virtual environment**

```bash
python -m venv venv 
venv\Scripts\activate     # Windows 
source venv/bin/activate  # Linux/macOS  
```

3.  **Install dependencies**

```bash
pip install -r requirements.txt
```

4.  **Create resources folder to store data**

```bash
mkdir resources  
```

Upload M-PESA statement in the resources folder as `mpesa_statement.pdf`.

5.  **Configure `.env`**

```bash
FILE_PASSWORD=your_pdf_password
GROQ_API_KEY=your_groq_api_key
```

6.  **Run the scripts to process the data**

```bash
python scripts\clean_mpesa_statement.py
python scripts\classify_transactions.py
```

Processed data will be stored in the resources folder as `unlocked_statement.pdf`, `clean_mpesa_transactions.csv`, `mpesa_classified.csv`

* * *

## Future Enhancements

## 

*   Integrate **budgeting**.
*   Enable **multi-file upload** for batch processing multiple statements.