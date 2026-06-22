import os
import uuid
import pandas as pd
import requests
import logging
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote

# ==================================================
# Configuration
# ==================================================
INPUT_CSV_FILES = ["merged_links_indiaCode.csv", "merged_links_legislative.csv"]
OUTPUT_CSV = "downloaded_pdfs_metadata.csv"
FAILED_CSV = "failed_links.csv"
BASE_DOWNLOAD_FOLDER = "PDF"
PDF_COLUMN = "pdf_link"

# ==================================================
# Logging Configuration
# ==================================================
LOG_FILE = "download_log.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("=== PDF Download Process Started ===")

# ==================================================
# Utilities
# ==================================================
def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)
    logging.info(f"Verified directory: {path}")

def extract_pdf_name(link: str) -> str:
    parsed = urlparse(link)
    query_params = parse_qs(parsed.query)
    if "rfilename" in query_params:
        file_name = query_params["rfilename"][0]
    elif "sfilename" in query_params:
        file_name = query_params["sfilename"][0]
    else:
        file_name = os.path.basename(parsed.path)
    file_name = unquote(file_name)
    if not file_name.endswith(".pdf"):
        file_name += ".pdf"
    return file_name

def append_row_to_csv(csv_file: str, data: dict):
    df = pd.DataFrame([data])
    exists = os.path.exists(csv_file)
    df.to_csv(csv_file, mode="a", index=False, header=not exists)

# ==================================================
# Core Download Function
# ==================================================
def download_pdf(session, link: str, dest_folder: str, source_file: str):
    pdf_id = str(uuid.uuid4())
    pdf_name = extract_pdf_name(link)
    local_path = os.path.join(dest_folder, pdf_name)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.indiacode.nic.in/",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    }

    try:
        response = session.get(link, headers=headers, timeout=40, stream=True, allow_redirects=True)

        if response.status_code != 200:
            logging.warning(f"Failed ({response.status_code}): {link}")
            append_row_to_csv(FAILED_CSV, {
                "pdf_url": link,
                "status": response.status_code,
                "reason": "HTTP error",
                "source_file": source_file
            })
            return False

        # Read the first chunk to verify file type
        chunk = next(response.iter_content(2048), b'')
        if not chunk.startswith(b"%PDF"):
            # Detect HTML / repealed pages
            html_sniff = chunk[:100].decode(errors="ignore").lower()
            if "repealed" in html_sniff or "<html" in html_sniff:
                logging.warning(f"Repealed or HTML page detected: {link}")
                append_row_to_csv(FAILED_CSV, {
                    "pdf_url": link,
                    "status": "HTML/Repealed",
                    "reason": "Not a real PDF",
                    "source_file": source_file
                })
                return False
            logging.warning(f"Invalid content (not PDF): {link}")
            return False

        # Save the PDF
        with open(local_path, "wb") as f:
            f.write(chunk)
            for data in response.iter_content(8192):
                if data:
                    f.write(data)

        # Save metadata
        metadata = {
            "pdf_id": pdf_id,
            "pdf_name": pdf_name,
            "pdf_url": link,
            "local_path": local_path,
            "source_file": source_file,
            "download_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        append_row_to_csv(OUTPUT_CSV, metadata)
        logging.info(f"Downloaded: {pdf_name}")
        return True

    except Exception as e:
        logging.error(f"Exception downloading {link}: {e}")
        append_row_to_csv(FAILED_CSV, {
            "pdf_url": link,
            "status": "Exception",
            "reason": str(e),
            "source_file": source_file
        })
        return False

# ==================================================
# CSV Processor
# ==================================================
def process_csv(csv_file: str):
    try:
        df = pd.read_csv(csv_file)
        if PDF_COLUMN not in df.columns:
            raise ValueError(f"Missing column '{PDF_COLUMN}' in {csv_file}")

        folder_name = os.path.splitext(os.path.basename(csv_file))[0]
        dest_folder = os.path.join(BASE_DOWNLOAD_FOLDER, folder_name)
        ensure_directory(dest_folder)

        logging.info(f"Processing: {csv_file} | Total links: {len(df)}")

        session = requests.Session()
        success = 0
        for link in df[PDF_COLUMN].dropna():
            link = str(link).strip()
            if not link:
                continue
            if download_pdf(session, link, dest_folder, csv_file):
                success += 1

        logging.info(f"Completed {csv_file}: {success} PDFs downloaded.")

    except Exception as e:
        logging.error(f"Error processing {csv_file}: {e}")

# ==================================================
# Main
# ==================================================
def main():
    ensure_directory(BASE_DOWNLOAD_FOLDER)
    for csv_file in INPUT_CSV_FILES:
        process_csv(csv_file)
    logging.info("=== PDF Download Process Completed ===")

if __name__ == "__main__":
    main()
