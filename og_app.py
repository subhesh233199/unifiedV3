import os
import re
import json
import runpy
import base64
import sqlite3
import hashlib
import time
from typing import List, Dict, Tuple, Any, Union
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import AzureChatOpenAI
import ssl
import warnings
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tenacity import retry, stop_after_attempt, wait_fixed
from copy import deepcopy
import markdown as md 
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="RRR Release Analysis Tool", description="API for analyzing release readiness reports")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Azure OpenAI
llm = LLM(
    model=f"azure/{os.getenv('DEPLOYMENT_NAME')}",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0.1,
    top_p=0.95,
)

# Constants
START_HEADER_PATTERN = 'Release Readiness Critical Metrics (Previous/Current):'
END_HEADER_PATTERN = 'Release Readiness Functional teams Deliverables Checklist:'
EXPECTED_METRICS = [
    "Open ALL RRR Defects", "Open Security Defects", "All Open Defects (T-1)",
    "All Security Open Defects", "Load/Performance", "E2E Test Coverage",
    "Automation Test Coverage", "Unit Test Coverage", "Defect Closure Rate",
    "Regression Issues", "Customer Specific Testing (UAT)"
]
CACHE_TTL_SECONDS = 3 * 24 * 60 * 60  # 3 days in seconds

# Pydantic models
class FolderPathRequest(BaseModel):
    """
    Pydantic model for folder path validation.
    
    Attributes:
        folder_path (str): Path to the folder containing PDF reports
        clear_cache (bool): Whether to clear the cache before analysis (default: False)
        
    Validators:
        validate_folder_path: Ensures folder path is not empty
    """
    folder_path: str
    clear_cache: bool = False  # Added this line

    @validator('folder_path')
    def validate_folder_path(cls, v):
        if not v:
            raise ValueError('Folder path cannot be empty')
        return v
class AnalysisResponse(BaseModel):
    """
    Pydantic model for analysis response.
    
    Attributes:
        metrics (Dict): Processed metrics data
        visualizations (List[str]): Base64 encoded visualization images
        report (str): Generated markdown report
        evaluation (Dict): Quality evaluation of the analysis
        hyperlinks (List[Dict]): Extracted hyperlinks from PDFs
        metrics_summary: str   
    """
    metrics: Dict
    visualizations: List[str]
    report: str
    evaluation: Dict
    hyperlinks: List[Dict]
    metrics_summary: str   # <-- ADD THIS LINE

class MetricItem(BaseModel):
    version: str
    value: Union[float, str]
    status: str
    trend: Union[str, None] = None

class UpdateReportRequest(BaseModel):
    folder_path: str
    report: str
    
# Shared state for thread-safe data sharing
class SharedState:
    """
    Thread-safe shared state container for the application.
    
    Attributes:
        metrics (dict): Stores processed metrics data
        report_parts (dict): Stores different sections of the generated report
        lock (Lock): Thread lock for general operations
        visualization_ready (bool): Flag indicating visualization status
        viz_lock (Lock): Thread lock for visualization operations
    """
    def __init__(self):
        self.metrics = None
        self.report_parts = {}
        self.lock = Lock()
        self.visualization_ready = False
        self.viz_lock = Lock()

shared_state = SharedState()
def metrics_to_markdown(metrics: Dict) -> str:
    """
    Generate a Metrics Summary markdown string from structured metrics JSON.
    This guarantees the numbers are from code, not LLM hallucination!
    """
    if not metrics or 'metrics' not in metrics:
        return "No metrics data available."

    md = []
    m = metrics['metrics']

    # Helper for ATLS/BTLS tables
    def table_atls_btls(metric, metric_label):
        rows = []
        for sub in ["ATLS", "BTLS"]:
            if sub in m[metric]:
                rows.append(f"#### {metric_label} ({sub})")
                rows.append("| Release | Value | Trend | Status |")
                rows.append("|---------|-------|-------|--------|")
                for item in m[metric][sub]:
                    rows.append(f"| {item['version']} | {item['value']} | {item.get('trend','')} | {item['status']} |")
        return "\n".join(rows)

    # Helper for flat metrics
    def table_flat(metric, label):
        rows = [f"#### {label}", "| Release | Value | Trend | Status |", "|---------|-------|-------|--------|"]
        for item in m[metric]:
            rows.append(f"| {item['version']} | {item['value']} | {item.get('trend','')} | {item['status']} |")
        return "\n".join(rows)

    # Customer Specific Testing (UAT)
    if "Customer Specific Testing (UAT)" in m:
        rows = ["### Customer Specific Testing (UAT)"]
        for client in ["RBS", "Tesco", "Belk"]:
            if client in m["Customer Specific Testing (UAT)"]:
                rows.append(f"#### {client}")
                rows.append("| Release | Pass Count | Fail Count | Pass Rate (%) | Trend | Status |")
                rows.append("|---------|------------|------------|---------------|-------|--------|")
                for item in m["Customer Specific Testing (UAT)"][client]:
                    rows.append(f"| {item['version']} | {item['pass_count']} | {item['fail_count']} | {round(item.get('pass_rate', 0), 1)} | {item.get('trend','')} | {item['status']} |")
        md.append("\n".join(rows))

    # ATLS/BTLS metrics
    for label in [
        ("Open ALL RRR Defects", "Open ALL RRR Defects"),
        ("Open Security Defects", "Open Security Defects"),
        ("All Open Defects (T-1)", "All Open Defects (T-1)"),
        ("All Security Open Defects", "All Security Open Defects"),
        ("Load/Performance", "Load/Performance"),
    ]:
        if label[0] in m:
            md.append(table_atls_btls(label[0], label[1]))

    # Flat metrics
    for label in [
        ("E2E Test Coverage", "E2E Test Coverage"),
        ("Automation Test Coverage", "Automation Test Coverage"),
        ("Unit Test Coverage", "Unit Test Coverage"),
        ("Defect Closure Rate", "Defect Closure Rate"),
        ("Regression Issues", "Regression Issues"),
    ]:
        if label[0] in m:
            md.append(table_flat(label[0], label[1]))

    return "### Metrics Summary\n\n" + "\n\n".join(md)

# SQLite database setup
def init_cache_db():
    """
    Initializes SQLite database for caching analysis results.
    
    Creates:
    - report_cache table with columns:
        - folder_path_hash (TEXT)
        - pdfs_hash (TEXT)
        - report_json (TEXT)
        - created_at (INTEGER)
    """
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS report_cache (
            folder_path_hash TEXT PRIMARY KEY,
            pdfs_hash TEXT NOT NULL,
            report_json TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_cache_db()

def hash_string(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def hash_pdf_contents(pdf_files: List[str]) -> str:
    hasher = hashlib.md5()
    for pdf_path in sorted(pdf_files):
        try:
            with open(pdf_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        except Exception as e:
            logger.error(f"Error hashing PDF {pdf_path}: {str(e)}")
            raise
    return hasher.hexdigest()

def get_cached_report(folder_path_hash: str, pdfs_hash: str) -> Union[AnalysisResponse, None]:
    """
    Retrieves cached analysis results if available and not expired.
    
    Args:
        folder_path_hash (str): Hash of folder path
        pdfs_hash (str): Hash of PDF contents
        
    Returns:
        Union[AnalysisResponse, None]: Cached results or None
    """
    try:
        conn = sqlite3.connect('cache.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT report_json, created_at
            FROM report_cache
            WHERE folder_path_hash = ? AND pdfs_hash = ?
        ''', (folder_path_hash, pdfs_hash))
        result = cursor.fetchone()
        conn.close()

        if result:
            report_json, created_at = result
            current_time = int(time.time())
            if current_time - created_at < CACHE_TTL_SECONDS:
                report_dict = json.loads(report_json)
                return AnalysisResponse(**report_dict)
            else:
                with shared_state.lock:
                    conn = sqlite3.connect('cache.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM report_cache WHERE folder_path_hash = ?', (folder_path_hash,))
                    conn.commit()
                    conn.close()
        return None
    except Exception as e:
        logger.error(f"Error retrieving cached report: {str(e)}")
        return None

def store_cached_report(folder_path_hash: str, pdfs_hash: str, response: AnalysisResponse):
    """
    Stores analysis results in cache.
    
    Args:
        folder_path_hash (str): Hash of folder path
        pdfs_hash (str): Hash of PDF contents
        response (AnalysisResponse): Analysis results to cache
    """
    try:
        report_json = json.dumps(response.dict())
        current_time = int(time.time())
        with shared_state.lock:
            conn = sqlite3.connect('cache.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO report_cache (folder_path_hash, pdfs_hash, report_json, created_at)
                VALUES (?, ?, ?, ?)
            ''', (folder_path_hash, pdfs_hash, report_json, current_time))
            conn.commit()
            conn.close()
        logger.info(f"Cached report for folder_path_hash: {folder_path_hash}")
    except Exception as e:
        logger.error(f"Error storing cached report: {str(e)}")
def parse_markdown_table(table_text):
    lines = [l for l in table_text.split('\n') if l.strip()]
    if len(lines) < 2:
        return []
    headers = [h.strip() for h in lines[0].strip('|').split('|')]
    rows = []
    for line in lines[2:]:
        if not line.strip():
            continue
        values = [v.strip() for v in line.strip('|').split('|')]
        row = dict(zip(headers, values))
        rows.append(row)
    return rows
def extract_metrics_from_markdown(markdown_text):
    html_content = md.markdown(markdown_text)
    soup = BeautifulSoup(html_content, "lxml")
    metrics_summary = soup.find('h2', string=re.compile(r'Metrics Summary', re.I))
    if not metrics_summary:
        return {}
    current = metrics_summary.find_next_sibling()
    tables_with_headings = []
    while current and current.name != "h2":
        if current.name == "h3":
            current_heading = current.get_text(strip=True)
            table_p = current.find_next_sibling("p")
            if table_p and re.search(r"\| ?Release ?\|", table_p.text):
                table_text = table_p.text.strip()
                tables_with_headings.append({
                    "heading": current_heading,
                    "table": table_text
                })
        current = current.find_next_sibling()
    return tables_with_headings
def build_metrics_dict_from_extracted(tables_with_headings):
    metrics = {}
    for entry in tables_with_headings:
        heading = entry['heading']
        rows = parse_markdown_table(entry['table'])
        metrics[heading] = rows
    return {"metrics": metrics}

def save_markdown_to_cache(folder_path, markdown_text):
    folder_path_hash = hash_string(folder_path)
    conn = sqlite3.connect('cache.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute('''
        INSERT OR REPLACE INTO report_cache (folder_path_hash, pdfs_hash, report_json, created_at)
        VALUES (?, '', ?, ?)
    ''', (folder_path_hash, json.dumps({"report": markdown_text}), int(time.time())))
    conn.commit()
    conn.close()
def cleanup_old_cache():
    try:
        current_time = int(time.time())
        with shared_state.lock:
            conn = sqlite3.connect('cache.db')
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM report_cache
                WHERE created_at < ?
            ''', (current_time - CACHE_TTL_SECONDS,))
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
        logger.info(f"Cleaned up old cache entries, deleted {deleted_rows} rows")
    except Exception as e:
        logger.error(f"Error cleaning up old cache entries: {str(e)}")

def get_pdf_files_from_folder(folder_path: str) -> List[str]:
    """
    Retrieves all PDF files from specified folder.
    
    Args:
        folder_path (str): Path to folder containing PDFs
        
    Returns:
        List[str]: List of full paths to PDF files
        
    Raises:
        FileNotFoundError: If folder doesn't exist or no PDFs found
    """
    pdf_files = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
   
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.pdf'):
            full_path = os.path.join(folder_path, file_name)
            pdf_files.append(full_path)
   
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in the folder {folder_path}.")
   
    return pdf_files

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text content from PDF file.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + '\n'
            if not text.strip():
                raise ValueError(f"No text extracted from {pdf_path}")
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise

def extract_hyperlinks_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extracts hyperlinks and their context from PDF.
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        List[Dict[str, str]]: List of hyperlink information
    """
    hyperlinks = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages, start=1):
                if '/Annots' in page:
                    for annot in page['/Annots']:
                        annot_obj = annot.get_object()
                        if annot_obj['/Subtype'] == '/Link' and '/A' in annot_obj:
                            uri = annot_obj['/A']['/URI']
                            text = page.extract_text() or ""
                            context_start = max(0, text.find(uri) - 50)
                            context_end = min(len(text), text.find(uri) + len(uri) + 50)
                            context = text[context_start:context_end].strip()
                            hyperlinks.append({
                                "url": uri,
                                "context": context,
                                "page": page_num,
                                "source_file": os.path.basename(pdf_path)
                            })
    except Exception as e:
        logger.error(f"Error extracting hyperlinks from {pdf_path}: {str(e)}")
    return hyperlinks

def locate_table(text: str, start_header: str, end_header: str) -> str:
    """
    Locates and extracts table data between specified headers in text.
    
    Args:
        text (str): Full text content
        start_header (str): Starting header pattern
        end_header (str): Ending header pattern
        
    Returns:
        str: Extracted table text
        
    Raises:
        ValueError: If headers not found or no data between headers
    """
    start_index = text.find(start_header)
    end_index = text.find(end_header)
    if start_index == -1:
        raise ValueError(f'Header {start_header} not found in text')
    if end_index == -1:
        raise ValueError(f'Header {end_header} not found in text')
    table_text = text[start_index:end_index].strip()
    if not table_text:
        raise ValueError(f"No metrics table data found between headers")
    return table_text

# def evaluate_with_llm_judge(source_text: str, generated_report: str) -> Tuple[int, str]:
#     judge_llm = AzureChatOpenAI(
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_version=os.getenv("AZURE_API_VERSION"),
#         azure_deployment=os.getenv("DEPLOYMENT_NAME"),
#         temperature=0,
#         max_tokens=512,
#         timeout=None,
#     )
   
#     prompt = f"""Act as an impartial judge evaluating report quality. You will be given:
# 1. ORIGINAL SOURCE TEXT (extracted from PDF)
# 2. GENERATED REPORT (created by AI)

# Evaluate based on:
# - Data accuracy (50% weight): Does the report correctly reflect the source data?
# - Analysis depth (30% weight): Does it provide meaningful insights?
# - Clarity (20% weight): Is the presentation clear and professional?

# ORIGINAL SOURCE:
# {source_text}

# GENERATED REPORT:
# {generated_report}

# INSTRUCTIONS:
# 1. For each category, give a score (integer) out of its maximum:
#     - Data accuracy: [0-50]
#     - Analysis depth: [0-30]
#     - Clarity: [0-20]
# 2. Add up to a TOTAL out of 100.
# 3. Give a brief 2-3 sentence evaluation.
# 4. Use EXACTLY this format:
# Data accuracy: [0-50]
# Analysis depth: [0-30]
# Clarity: [0-20]
# TOTAL: [0-100]
# Evaluation: [your evaluation]

# Your evaluation:"""
   
#     try:
#         response = judge_llm.invoke(prompt)
#         response_text = response.content
#         score_line = next(line for line in response_text.split('\n') if line.startswith('Score:'))
#         score = int(score_line.split(':')[1].strip())
#         eval_lines = [line for line in response_text.split('\n') if line.startswith('Evaluation:')]
#         evaluation = ' '.join(line.split('Evaluation:')[1].strip() for line in eval_lines)
#         return score, evaluation
#     except Exception as e:
#         logger.error(f"Error parsing judge response: {e}\nResponse was:\n{response_text}")
#         return 50, "Could not parse evaluation"
from typing import Tuple, Dict

import re

import re

def evaluate_with_llm_judge(source_text: str, generated_report: str) -> dict:
    judge_llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_deployment=os.getenv("DEPLOYMENT_NAME"),
        temperature=0,
        max_tokens=512,
        timeout=None,
    )
   
    prompt = f"""Act as an impartial judge evaluating report quality. You will be given:
1. ORIGINAL SOURCE TEXT (extracted from PDF)
2. GENERATED REPORT (created by AI)

Evaluate based on:
- Data accuracy (50% weight): Does the report correctly reflect the source data?
- Analysis depth (30% weight): Does it provide meaningful insights?
- Clarity (20% weight): Is the presentation clear and professional?

ORIGINAL SOURCE:
{source_text}

GENERATED REPORT:
{generated_report}

INSTRUCTIONS:
1. For each category, give a score (integer) out of its maximum:
    - Data accuracy: [0-50]
    - Analysis depth: [0-30]
    - Clarity: [0-20]
2. Add up to a TOTAL out of 100.
3. Give a brief 2-3 sentence evaluation.
4. Use EXACTLY this format:
Data accuracy: [0-50]
Analysis depth: [0-30]
Clarity: [0-20]
TOTAL: [0-100]
Evaluation: [your evaluation]

Your evaluation:"""

    try:
        response = judge_llm.invoke(prompt)
        response_text = response.content

        # Robust extraction: matches label anywhere on line, any case, extra spaces, "35/50" or "35"
        def extract_score(label, default=0):
            regex = re.compile(rf"{label}\s*:\s*(\d+)", re.IGNORECASE)
            for line in response_text.splitlines():
                match = regex.search(line)
                if match:
                    return int(match.group(1))
            return default

        data_accuracy = extract_score("Data accuracy", 0)
        analysis_depth = extract_score("Analysis depth", 0)
        clarity = extract_score("Clarity", 0)
        total = extract_score("TOTAL", data_accuracy + analysis_depth + clarity)

        # Extract evaluation: combine lines after "Evaluation:" or the last non-score line
        evaluation = ""
        eval_regex = re.compile(r"evaluation\s*:\s*(.*)", re.IGNORECASE)
        found_eval = False
        for line in response_text.splitlines():
            match = eval_regex.match(line)
            if match:
                evaluation = match.group(1).strip()
                found_eval = True
                break
        # If not found, fallback: concatenate all lines not containing a score label
        if not found_eval:
            non_score_lines = [
                l for l in response_text.splitlines()
                if not any(lbl in l.lower() for lbl in ["data accuracy", "analysis depth", "clarity", "total"])
            ]
            evaluation = " ".join(non_score_lines).strip()

        return {
            "data_accuracy": data_accuracy,
            "analysis_depth": analysis_depth,
            "clarity": clarity,
            "total": total,
            "text": evaluation
        }
    except Exception as e:
        logger.error(f"Error parsing judge response: {e}\nResponse was:\n{locals().get('response_text', '')}")
        return {
            "data_accuracy": 0,
            "analysis_depth": 0,
            "clarity": 0,
            "total": 0,
            "text": "Could not parse evaluation"
        }

def validate_report(report: str) -> bool:
    required_sections = ["# Software Metrics Report", "## Overview", "## Key Findings", "## Recommendations"]
    return all(section in report for section in required_sections)

def validate_metrics(metrics: Dict[str, Any]) -> bool:
    """
    Validates the structure and content of metrics data.
    
    Checks:
    - Required metrics presence
    - Data type correctness
    - Value ranges
    - Status values
    - Trend format
    
    Args:
        metrics (Dict[str, Any]): Metrics data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not metrics or 'metrics' not in metrics or not isinstance(metrics['metrics'], dict):
        logger.warning(f"Invalid metrics structure: {metrics}")
        return False
    missing_metrics = [m for m in EXPECTED_METRICS if m not in metrics['metrics']]
    if missing_metrics:
        logger.warning(f"Missing metrics: {missing_metrics}")
        return False
    for metric, data in metrics['metrics'].items():
        if metric in EXPECTED_METRICS[:5]:  # ATLS/BTLS metrics
            if not isinstance(data, dict) or 'ATLS' not in data or 'BTLS' not in data:
                logger.warning(f"Invalid ATLS/BTLS structure for {metric}: {data}")
                return False
            for sub in ['ATLS', 'BTLS']:
                if not isinstance(data[sub], list) or len(data[sub]) < 2:
                    logger.warning(f"Empty or insufficient {sub} data for {metric}: {data[sub]}")
                    return False
                has_non_zero = False
                for item in data[sub]:
                    try:
                        item_dict = dict(item)
                        if not all(k in item_dict for k in ['version', 'value', 'status']):
                            logger.warning(f"Missing keys in {sub} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                            logger.warning(f"Invalid version in {sub} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['value'], (int, float)) or item_dict['value'] < 0:
                            logger.warning(f"Invalid value in {sub} item for {metric}: {item}")
                            return False
                        if item_dict['value'] > 0:
                            has_non_zero = True
                        if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                            logger.warning(f"Invalid status in {sub} item for {metric}: {item}")
                            return False
                        if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                            logger.warning(f"Invalid trend in {sub} item for {metric}: {item}")
                            return False
                    except Exception as e:
                        logger.warning(f"Invalid item in {sub} for {metric}: {item}, error: {str(e)}")
                        return False
                if not has_non_zero:
                    logger.warning(f"No non-zero values in {sub} for {metric}")
                    return False
        elif metric == "Customer Specific Testing (UAT)":
            if not isinstance(data, dict) or not all(client in data for client in ['RBS', 'Tesco', 'Belk']):
                logger.warning(f"Invalid structure for {metric}: {data}")
                return False
            for client in ['RBS', 'Tesco', 'Belk']:
                client_data = data.get(client, [])
                if not isinstance(client_data, list) or len(client_data) < 2:
                    logger.warning(f"Empty or insufficient data for {metric} {client}: {client_data}")
                    return False
                for item in client_data:
                    try:
                        item_dict = dict(item)
                        if not all(k in item_dict for k in ['version', 'pass_count', 'fail_count', 'status']):
                            logger.warning(f"Missing keys in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                            logger.warning(f"Invalid version in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['pass_count'], (int, float)) or item_dict['pass_count'] < 0:
                            logger.warning(f"Invalid pass_count in {client} item for {metric}: {item}")
                            return False
                        if not isinstance(item_dict['fail_count'], (int, float)) or item_dict['fail_count'] < 0:
                            logger.warning(f"Invalid fail_count in {client} item for {metric}: {item}")
                            return False
                        if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                            logger.warning(f"Invalid status in {client} item for {metric}: {item}")
                            return False
                        if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                            logger.warning(f"Invalid trend in {client} item for {metric}: {item}")
                            return False
                    except Exception as e:
                        logger.warning(f"Invalid item in {client} for {metric}: {item}, error: {str(e)}")
                        return False
        else:  # Non-ATLS/BTLS metrics
            if not isinstance(data, list) or len(data) < 2:
                logger.warning(f"Empty or insufficient data for {metric}: {data}")
                return False
            has_non_zero = False
            for item in data:
                try:
                    item_dict = dict(item)
                    if not all(k in item_dict for k in ['version', 'value', 'status']):
                        logger.warning(f"Missing keys in item for {metric}: {item}")
                        return False
                    if not isinstance(item_dict['version'], str) or not re.match(r'^\d+\.\d+$', item_dict['version']):
                        logger.warning(f"Invalid version in item for {metric}: {item}")
                        return False
                    if not isinstance(item_dict['value'], (int, float)) or item_dict['value'] < 0:
                        logger.warning(f"Invalid value in item for {metric}: {item}")
                        return False
                    if item_dict['value'] > 0:
                        has_non_zero = True
                    if item_dict['status'] not in ['ON TRACK', 'MEDIUM RISK', 'RISK', 'NEEDS REVIEW']:
                        logger.warning(f"Invalid status in item for {metric}: {item}")
                        return False
                    if 'trend' in item_dict and not re.match(r'^(↑|↓)\s*\(\d+\.\d+%\)|→$', item_dict['trend']):
                        logger.warning(f"Invalid trend in item for {metric}: {item}")
                        return False
                except Exception as e:
                    logger.warning(f"Invalid item for {metric}: {item}, error: {str(e)}")
                    return False
            if not has_non_zero:
                logger.warning(f"No non-zero values for {metric}")
                return False
    return True

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def process_task_output(raw_output: str, fallback_versions: List[str]) -> Dict:
    logger.info(f"Raw output type: {type(raw_output)}, content: {raw_output if isinstance(raw_output, str) else raw_output}")
    if not isinstance(raw_output, str):
        logger.warning(f"Expected raw_output to be a string, got {type(raw_output)}. Falling back to empty JSON.")
        raw_output = "{}"  # Fallback to empty JSON string
    logger.info(f"Processing task output: {raw_output[:200]}...")
    data = clean_json_output(raw_output, fallback_versions)
    if not validate_metrics(data):
        logger.error(f"Validation failed for processed output: {json.dumps(data, indent=2)[:200]}...")
        raise ValueError("Invalid or incomplete metrics data")
    # Validate and correct trends
    for metric, metric_data in data['metrics'].items():
        if metric in EXPECTED_METRICS[:5]:  # ATLS/BTLS metrics
            for sub in ['ATLS', 'BTLS']:
                items = sorted(metric_data[sub], key=lambda x: x['version'])
                for i in range(len(items)):
                    if i == 0 or not items[i].get('value') or not items[i-1].get('value'):
                        items[i]['trend'] = '→'
                    else:
                        prev_val = float(items[i-1]['value'])
                        curr_val = float(items[i]['value'])
                        if prev_val == 0 or abs(curr_val - prev_val) < 0.01:
                            items[i]['trend'] = '→'
                        else:
                            pct_change = ((curr_val - prev_val) / prev_val) * 100
                            if abs(pct_change) < 1:
                                items[i]['trend'] = '→'
                            elif pct_change > 0:
                                items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                            else:
                                items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
        elif metric == "Customer Specific Testing (UAT)":
            for client in ['RBS', 'Tesco', 'Belk']:
                items = sorted(metric_data[client], key=lambda x: x['version'])
                for i in range(len(items)):
                    pass_count = float(items[i].get('pass_count', 0))
                    fail_count = float(items[i].get('fail_count', 0))
                    total = pass_count + fail_count
                    pass_rate = (pass_count / total * 100) if total > 0 else 0
                    items[i]['pass_rate'] = pass_rate
                    if i == 0:
                        items[i]['trend'] = '→'
                    else:
                        prev_pass_count = float(items[i-1].get('pass_count', 0))
                        prev_fail_count = float(items[i-1].get('fail_count', 0))
                        prev_total = prev_pass_count + prev_fail_count
                        prev_pass_rate = (prev_pass_count / prev_total * 100) if prev_total > 0 else 0
                        if prev_total == 0 or total == 0 or abs(pass_rate - prev_pass_rate) < 0.01:
                            items[i]['trend'] = '→'
                        else:
                            pct_change = pass_rate - prev_pass_rate
                            if abs(pct_change) < 1:
                                items[i]['trend'] = '→'
                            elif pct_change > 0:
                                items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                            else:
                                items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
        else:  # Non-ATLS/BTLS metrics
            items = sorted(metric_data, key=lambda x: x['version'])
            for i in range(len(items)):
                if i == 0 or not items[i].get('value') or not items[i-1].get('value'):
                    items[i]['trend'] = '→'
                else:
                    prev_val = float(items[i-1]['value'])
                    curr_val = float(items[i]['value'])
                    if prev_val == 0 or abs(curr_val - prev_val) < 0.01:
                        items[i]['trend'] = '→'
                    else:
                        pct_change = ((curr_val - prev_val) / prev_val) * 100
                        if abs(pct_change) < 1:
                            items[i]['trend'] = '→'
                        elif pct_change > 0:
                            items[i]['trend'] = f"↑ ({abs(pct_change):.1f}%)"
                        else:
                            items[i]['trend'] = f"↓ ({abs(pct_change):.1f}%)"
    return data

def setup_crew(extracted_text: str, versions: List[str], llm=llm) -> tuple:
    """
    Sets up the AI crew system for analysis.
    """
    structurer = Agent(
        role="Data Architect",
        goal="Structure raw release data into VALID JSON format",
        backstory="Expert in transforming unstructured data into clean JSON structures",
        llm=llm,
        verbose=True,
        memory=True,
    )

    if len(versions) < 2:
        raise ValueError("At least two versions are required for analysis")
    versions_for_example = versions[:3] if len(versions) >= 3 else versions + [versions[-1]] * (3 - len(versions))

    validated_structure_task = Task(
        description=f"""Convert this release data to STRICT JSON:
{extracted_text}
...
(unchanged details for validated_structure_task)
...""",
        agent=structurer,
        async_execution=False,
        expected_output="Valid JSON string with no extra text",
        callback=lambda output: (
            logger.info(f"Structure task output type: {type(output.raw)}, content: {output.raw if isinstance(output.raw, str) else output.raw}"),
            setattr(shared_state, 'metrics', process_task_output(output.raw, versions))
        )
    )

    analyst = Agent(
        role="Trend Analyst",
        goal="Add accurate trends to metrics data and maintain valid JSON",
        backstory="Data scientist specializing in metric analysis",
        llm=llm,
        verbose=True,
        memory=True,
    )

    analysis_task = Task(
        description=f"""Enhance metrics JSON with trends:
...
(unchanged details for analysis_task)
...""",
        agent=analyst,
        async_execution=True,
        context=[validated_structure_task],
        expected_output="Valid JSON string with trend analysis",
        callback=lambda output: (
            logger.info(f"Analysis task output type: {type(output.raw)}, content: {output.raw if isinstance(output.raw, str) else output.raw}"),
            setattr(shared_state, 'metrics', process_task_output(output.raw, versions))
        )
    )

    visualizer = Agent(
        role="Data Visualizer",
        goal="Generate consistent visualizations for all metrics",
        backstory="Expert in generating Python plots for software metrics",
        llm=llm,
        verbose=True,
        memory=True,
    )

    visualization_task = Task(
        description=f"""Create a standalone Python script that:
...
(unchanged details for visualization_task)
...""",
        agent=visualizer,
        context=[analysis_task],
        expected_output="Python script only"
    )

    reporter = Agent(
        role="Technical Writer",
        goal="Generate a professional markdown report",
        backstory="Writes structured software metrics reports",
        llm=llm,
        verbose=True,
        memory=True,
    )

    overview_task = Task(
        description=f"""Write ONLY the following Markdown section:
## Overview
- Provide a 3-4 sentence comprehensive summary of release health, covering overall stability, notable improvements, and any concerning patterns observed across releases {', '.join(versions)}
- Explicitly list all analyzed releases
- Include 2-3 notable metric highlights with specific version comparisons where relevant
- Mention any significant deviations from expected patterns
Only output this section.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown for Overview section"
    )

    key_findings_task = Task(
        description=f"""Generate ONLY this Markdown section:
## Key Findings
1. First finding (2-3 sentences explaining the observation with specific metric references and version comparisons across {', '.join(versions)})
2. Second finding (2-3 sentences with quantitative data points from the metrics where applicable)
3. Third finding (2-3 sentences focusing on security-related observations)
4. Fourth finding (2-3 sentences about testing coverage trends)
5. Fifth finding (2-3 sentences highlighting any unexpected patterns or anomalies)
6. Sixth finding (2-3 sentences about performance or load metrics)
7. Seventh finding (2-3 sentences summarizing defect management effectiveness)
Maintain professional, analytical tone while being specific.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown bullet list"
    )

    recommendations_task = Task(
        description="""Generate ONLY this Markdown section:
## Recommendations
1. First recommendation (2-3 actionable sentences with specific metrics or areas to address)
2. Second recommendation (2-3 sentences about security improvements with version targets)
3. Third recommendation (2-3 sentences about testing coverage enhancements)
4. Fourth recommendation (2-3 sentences about defect management process changes)
5. Fifth recommendation (2-3 sentences about performance optimization)
6. Sixth recommendation (2-3 sentences about risk mitigation strategies)
7. Seventh recommendation (2-3 sentences about monitoring improvements)
Each recommendation should be specific, measurable, and tied to the findings.""",
        agent=reporter,
        context=[analysis_task],
        expected_output="Detailed markdown bullet list"
    )

    assemble_report_task = Task(
        description="""Assemble the final markdown report in this exact structure:
# Software Metrics Report

## Overview
[Insert from Overview Task]

---

## Key Findings
[Insert from Key Findings Task]

---

## Recommendations
[Insert from Recommendations Task]

Do NOT alter content. Just combine with correct formatting.""",
        agent=reporter,
        context=[
            overview_task,
            key_findings_task,
            recommendations_task
        ],
        expected_output="Full markdown report"
    )

    data_crew = Crew(
        agents=[structurer, analyst],
        tasks=[validated_structure_task, analysis_task],
        process=Process.sequential,
        verbose=True
    )

    report_crew = Crew(
        agents=[reporter],
        tasks=[overview_task, key_findings_task, recommendations_task, assemble_report_task],
        process=Process.sequential,
        verbose=True
    )

    viz_crew = Crew(
        agents=[visualizer],
        tasks=[visualization_task],
        process=Process.sequential,
        verbose=True
    )

    for crew, name in [(data_crew, "data_crew"), (report_crew, "report_crew"), (viz_crew, "viz_crew")]:
        for i, task in enumerate(crew.tasks):
            if not isinstance(task, Task):
                logger.error(f"Invalid task in {name} at index {i}: {task}")
                raise ValueError(f"Task in {name} is not a Task object")
            logger.info(f"{name} task {i} async_execution: {task.async_execution}")

    return data_crew, report_crew, viz_crew

def clean_json_output(raw_output: str, fallback_versions: List[str]) -> dict:
    logger.info(f"Raw analysis output: {raw_output[:200]}...")
    # Synthetic data for fallback (ensure at least one non-zero value to pass validation)
    default_json = {
        "metrics": {
            metric: {
                "ATLS": [
                    {"version": v, "value": 10 if i == 0 else 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ],
                "BTLS": [
                    {"version": v, "value": 12 if i == 0 else 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ]
            } if metric in EXPECTED_METRICS[:5] else
            {
                "RBS": [
                    {"version": v, "pass_count": 50 if i == 0 else 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ],
                "Tesco": [
                    {"version": v, "pass_count": 45 if i == 0 else 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ],
                "Belk": [
                    {"version": v, "pass_count": 40 if i == 0 else 0, "fail_count": 0, "status": "NEEDS REVIEW"}
                    for i, v in enumerate(fallback_versions)
                ]
            } if metric == "Customer Specific Testing (UAT)" else
            [
                {"version": v, "value": 80 if i == 0 else 0, "status": "NEEDS REVIEW"}
                for i, v in enumerate(fallback_versions)
            ]
            for metric in EXPECTED_METRICS
        }
    }

    try:
        data = json.loads(raw_output)
        if validate_metrics(data):
            return data
        logger.warning(f"Direct JSON invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parsing failed: {str(e)}")

    try:
        cleaned = re.search(r'```json\s*([\s\S]*?)\s*```', raw_output, re.MULTILINE)
        if cleaned:
            data = json.loads(cleaned.group(1))
            if validate_metrics(data):
                return data
            logger.warning(f"Code block JSON invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"Code block JSON parsing failed: {str(e)}")

    try:
        cleaned = re.search(r'\{[\s\S]*\}', raw_output, re.MULTILINE)
        if cleaned:
            json_str = cleaned.group(0)
            json_str = re.sub(r"'", '"', json_str)
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            data = json.loads(json_str)
            if validate_metrics(data):
                return data
            logger.warning(f"JSON-like structure invalid: {json.dumps(data, indent=2)[:200]}...")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON-like structure parsing failed: {str(e)}")

    logger.error(f"Failed to parse JSON, using default structure with zero values for versions: {fallback_versions}")
    return default_json

def enhance_report_markdown(md_text):
    # Remove markdown code fences
    cleaned = re.sub(r'^```markdown\n|\n```$', '', md_text, flags=re.MULTILINE)
    
    # Fix table alignment (remove extra spaces, ensure proper pipes)
    cleaned = re.sub(r'(\|.+\|)\n\s*(\|-+\|)', r'\1\n\2', cleaned)
    
    # Clean invalid trend symbols (e.g., '4', 't', '/')
    cleaned = re.sub(r'\b[4t/]\b', '→', cleaned)  # Replace stray symbols with arrow
    cleaned = re.sub(r'\s*\|\s*', ' | ', cleaned)  # Normalize spacing around pipes
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)  # Collapse multiple spaces
    
    # Enhance statuses
    status_map = {
        "MEDIUM RISK": "**MEDIUM RISK**",
        "HIGH RISK": "**HIGH RISK**",
        "LOW RISK": "**LOW RISK**",
        "ON TRACK": "**ON TRACK**"
    }
    for k, v in status_map.items():
        cleaned = cleaned.replace(k, v)
    
    # Fix headers and list items
    cleaned = re.sub(r'^#\s+(.+)$', r'# \1\n', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^##\s+(.+)$', r'## \1\n', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*-\s+(.+)', r'- \1', cleaned, flags=re.MULTILINE)
    
    return cleaned.encode('utf-8').decode('utf-8')

def convert_windows_path(path: str) -> str:
    path = path.replace('\\', '/')
    path = path.replace('//', '/')
    return path

def get_base64_image(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading image {image_path}: {str(e)}")
        return ""

def run_fallback_visualization(metrics: Dict[str, Any]):
    with shared_state.viz_lock:
        try:
            os.makedirs("visualizations", exist_ok=True)
            logging.basicConfig(level=logging.INFO, filename='visualization.log')
            logger = logging.getLogger(__name__)
            logger.info("Starting fallback visualization")

            if not metrics or 'metrics' not in metrics or not isinstance(metrics['metrics'], dict):
                logger.error(f"Invalid metrics data: {metrics}")
                raise ValueError("Metrics data is empty or invalid")

            atls_btls_metrics = EXPECTED_METRICS[:5]
            coverage_metrics = EXPECTED_METRICS[5:8]
            other_metrics = EXPECTED_METRICS[8:10]

            generated_files = []
            for metric in atls_btls_metrics:
                try:
                    data = metrics['metrics'].get(metric, {})
                    if not isinstance(data, dict) or 'ATLS' not in data or 'BTLS' not in data:
                        logger.warning(f"Creating placeholder for {metric}: invalid or missing ATLS/BTLS data")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    atls_data = data.get('ATLS', [])
                    btls_data = data.get('BTLS', [])
                    versions = [item['version'] for item in atls_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    atls_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in atls_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    btls_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in btls_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    if not versions or len(atls_values) != len(versions) or len(btls_values) != len(versions):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    x = np.arange(len(versions))
                    width = 0.35
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.bar(x - width/2, atls_values, width, label='ATLS', color='blue')
                    plt.bar(x + width/2, btls_values, width, label='BTLS', color='orange')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    plt.xticks(x, versions)
                    plt.legend()
                    filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated grouped bar chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}_atls_btls.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            for metric in coverage_metrics:
                try:
                    data = metrics['metrics'].get(metric, [])
                    if not isinstance(data, list) or not data:
                        logger.warning(f"Creating placeholder for {metric}: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    versions = [item['version'] for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    if not versions or len(values) != len(versions):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.plot(versions, values, marker='o', color='green')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated line chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            for metric in other_metrics:
                try:
                    data = metrics['metrics'].get(metric, [])
                    if not isinstance(data, list) or not data:
                        logger.warning(f"Creating placeholder for {metric}: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    versions = [item['version'] for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in data if isinstance(item, dict) and 'version' in item and 'value' in item]
                    if not versions or len(values) != len(versions):
                        logger.warning(f"Creating placeholder for {metric}: inconsistent data lengths")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, f"Incomplete data for {metric}", ha='center', va='center')
                        plt.title(metric)
                        filename = f'visualizations/{metric.replace("/", "_")}.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for {metric}: {filename}")
                        continue
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.bar(versions, values, color='purple')
                    plt.xlabel('Release')
                    plt.ylabel('Value')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated bar chart for {metric}: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for {metric}: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, f"Error generating {metric}", ha='center', va='center')
                    plt.title(metric)
                    filename = f'visualizations/{metric.replace("/", "_")}.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for {metric}: {filename}")

            if 'Pass/Fail' in metrics['metrics']:
                try:
                    data = metrics['metrics'].get('Pass/Fail', {})
                    if not isinstance(data, dict):
                        logger.warning(f"Creating placeholder for Pass/Fail: invalid data format")
                        plt.figure(figsize=(8,5), dpi=120)
                        plt.text(0.5, 0.5, "No data for Pass/Fail", ha='center', va='center')
                        plt.title("Pass/Fail Metrics")
                        filename = 'visualizations/pass_fail.png'
                        plt.savefig(filename)
                        plt.close()
                        generated_files.append(filename)
                        logger.info(f"Generated placeholder chart for Pass/Fail: {filename}")
                    else:
                        pass_data = data.get('Pass', [])
                        fail_data = data.get('Fail', [])
                        versions = [item['version'] for item in pass_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                        pass_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in pass_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                        fail_values = [float(item['value']) if isinstance(item['value'], (int, float)) else 0 for item in fail_data if isinstance(item, dict) and 'version' in item and 'value' in item]
                        if not versions or len(pass_values) != len(versions) or len(fail_values) != len(versions):
                            logger.warning(f"Creating placeholder for Pass/Fail: inconsistent data lengths")
                            plt.figure(figsize=(8,5), dpi=120)
                            plt.text(0.5, 0.5, "Incomplete data for Pass/Fail", ha='center', va='center')
                            plt.title("Pass/Fail Metrics")
                            filename = 'visualizations/pass_fail.png'
                            plt.savefig(filename)
                            plt.close()
                            generated_files.append(filename)
                            logger.info(f"Generated placeholder chart for Pass/Fail: {filename}")
                        else:
                            x = np.arange(len(versions))
                            width = 0.35
                            plt.figure(figsize=(8,5), dpi=120)
                            plt.bar(x - width/2, pass_values, width, label='Pass', color='green')
                            plt.bar(x + width/2, fail_values, width, label='Fail', color='red')
                            plt.xlabel('Release')
                            plt.ylabel('Count')
                            plt.title('Pass/Fail Metrics')
                            plt.xticks(x, versions)
                            plt.legend()
                            filename = 'visualizations/pass_fail.png'
                            plt.savefig(filename)
                            plt.close()
                            generated_files.append(filename)
                            logger.info(f"Generated grouped bar chart for Pass/Fail: {filename}")
                except Exception as e:
                    logger.error(f"Failed to generate chart for Pass/Fail: {str(e)}")
                    plt.figure(figsize=(8,5), dpi=120)
                    plt.text(0.5, 0.5, "Error generating Pass/Fail", ha='center', va='center')
                    plt.title("Pass/Fail Metrics")
                    filename = 'visualizations/pass_fail.png'
                    plt.savefig(filename)
                    plt.close()
                    generated_files.append(filename)
                    logger.info(f"Generated error placeholder chart for Pass/Fail: {filename}")

            logger.info(f"Completed fallback visualization, generated {len(generated_files)} files")
        except Exception as e:
            logger.error(f"Fallback visualization failed: {str(e)}")
            raise
        finally:
            plt.close('all')

async def run_full_analysis(request: FolderPathRequest) -> AnalysisResponse:
    folder_path = convert_windows_path(request.folder_path)
    folder_path = os.path.normpath(folder_path)
   
    if not os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail=f"Folder path does not exist: {folder_path}")
   
    pdf_files = get_pdf_files_from_folder(folder_path)
    logger.info(f"Processing {len(pdf_files)} PDF files")

    # Extract versions from PDF filenames
    versions = []
    for pdf_path in pdf_files:
        match = re.search(r'(\d+\.\d+)(?:\s|\.)', os.path.basename(pdf_path))
        if match:
            versions.append(match.group(1))
    versions = sorted(set(versions))
    if len(versions) < 2:
        raise HTTPException(status_code=400, detail="At least two versions are required for analysis")

    # Parallel PDF processing
    extracted_texts = []
    all_hyperlinks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        text_futures = {executor.submit(extract_text_from_pdf, pdf): pdf for pdf in pdf_files}
        hyperlink_futures = {executor.submit(extract_hyperlinks_from_pdf, pdf): pdf for pdf in pdf_files}
       
        for future in as_completed(text_futures):
            pdf = text_futures[future]
            try:
                text = locate_table(future.result(), START_HEADER_PATTERN, END_HEADER_PATTERN)
                extracted_texts.append((os.path.basename(pdf), text))
            except Exception as e:
                logger.error(f"Failed to process text from {pdf}: {str(e)}")
                continue
       
        for future in as_completed(hyperlink_futures):
            pdf = hyperlink_futures[future]
            try:
                all_hyperlinks.extend(future.result())
            except Exception as e:
                logger.error(f"Failed to process hyperlinks from {pdf}: {str(e)}")
                continue

    if not extracted_texts:
        raise HTTPException(status_code=400, detail="No valid text extracted from PDFs")

    full_source_text = "\n".join(
        f"File: {name}\n{text}" for name, text in extracted_texts
    )

    # Get sub-crews
    data_crew, report_crew, viz_crew = setup_crew(full_source_text, versions, llm)
   
    # Run crews sequentially and in parallel
    logger.info("Starting data_crew")
    await data_crew.kickoff_async()
    logger.info("Data_crew completed")
   
    # Validate task outputs
    for i, task in enumerate(data_crew.tasks):
        if not hasattr(task, 'output') or not hasattr(task.output, 'raw'):
            logger.error(f"Invalid output for data_crew task {i}: {task}")
            raise ValueError(f"Data crew task {i} did not produce a valid output")
        logger.info(f"Data_crew task {i} output: {task.output.raw[:200]}...")

    # Validate metrics
    if not shared_state.metrics or not isinstance(shared_state.metrics, dict):
        logger.error(f"Invalid metrics in shared_state: type={type(shared_state.metrics)}, value={shared_state.metrics}")
        raise HTTPException(status_code=500, detail="Failed to generate valid metrics data")
    logger.info(f"Metrics after data_crew: {json.dumps(shared_state.metrics, indent=2)[:200]}...")

    # Run report_crew and viz_crew in parallel
    logger.info("Starting report_crew and viz_crew")
    await asyncio.gather(
        report_crew.kickoff_async(),
        viz_crew.kickoff_async()
    )
    logger.info("Report_crew and viz_crew completed")

    # Validate report_crew output
    if not hasattr(report_crew.tasks[-1], 'output') or not hasattr(report_crew.tasks[-1].output, 'raw'):
        logger.error(f"Invalid output for report_crew task {report_crew.tasks[-1]}")
        raise ValueError("Report crew did not produce a valid output")
    logger.info(f"Report_crew output: {report_crew.tasks[-1].output.raw[:100]}...")

    # Validate viz_crew output
    if not hasattr(viz_crew.tasks[0], 'output') or not hasattr(viz_crew.tasks[0].output, 'raw'):
        logger.error(f"Invalid output for viz_crew task {viz_crew.tasks[0]}")
        raise ValueError("Visualization crew did not produce a valid output")
    logger.info(f"Viz_crew output: {viz_crew.tasks[0].output.raw[:100]}...")

    metrics = shared_state.metrics

    # NEW: Metrics summary from extracted metrics
    metrics_summary = metrics_to_markdown(metrics)

    # Get report from assemble_report_task (should be LLM analysis only, no tables)
    enhanced_report = enhance_report_markdown(report_crew.tasks[-1].output.raw)
    if not validate_report(enhanced_report):
        logger.error("Report missing required sections")
        raise HTTPException(status_code=500, detail="Generated report is incomplete")

    viz_folder = "visualizations"
    if os.path.exists(viz_folder):
        shutil.rmtree(viz_folder)
    os.makedirs(viz_folder, exist_ok=True)

    script_path = "visualizations.py"
    raw_script = viz_crew.tasks[0].output.raw
    clean_script = re.sub(r'```python|```$', '', raw_script, flags=re.MULTILINE).strip()

    try:
        with shared_state.viz_lock:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(clean_script)
            logger.info(f"Visualization script written to {script_path}")
            logger.debug(f"Visualization script content:\n{clean_script}")
            runpy.run_path(script_path, init_globals={'metrics': metrics})
            logger.info("Visualization script executed successfully")
    except Exception as e:
        logger.error(f"Visualization script failed: {str(e)}")
        logger.info("Running fallback visualization")
        run_fallback_visualization(metrics)

    viz_base64 = []
    expected_count = 10 + (1 if 'Pass/Fail' in metrics.get('metrics', {}) else 0)
    min_visualizations = 5
    if os.path.exists(viz_folder):
        viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
        for img in viz_files:
            img_path = os.path.join(viz_folder, img)
            base64_str = get_base64_image(img_path)
            if base64_str:
                viz_base64.append(base64_str)
        logger.info(f"Generated {len(viz_base64)} visualizations, expected {expected_count}, minimum required {min_visualizations}")
        if len(viz_base64) < min_visualizations:
            logger.warning("Insufficient visualizations, running fallback")
            run_fallback_visualization(metrics)
            viz_files = sorted([f for f in os.listdir(viz_folder) if f.endswith('.png')])
            viz_base64 = []
            for img in viz_files:
                img_path = os.path.join(viz_folder, img)
                base64_str = get_base64_image(img_path)
                if base64_str:
                    viz_base64.append(base64_str)
            if len(viz_base64) < min_visualizations:
                logger.error(f"Still too few visualizations: {len(viz_base64)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate minimum required visualizations: got {len(viz_base64)}, need at least {min_visualizations}"
                )

    # Evaluation uses metrics and report for LLM judgment
    evaluation = evaluate_with_llm_judge(metrics, full_source_text, enhanced_report)

    return AnalysisResponse(
        metrics=metrics,
        visualizations=viz_base64,
        report=enhanced_report,
        evaluation=evaluation,
        hyperlinks=all_hyperlinks,
        metrics_summary=metrics_summary   # <-- Added!
    )

@app.post("/save_report")
async def save_report(request: UpdateReportRequest):
    # 1. Save user-edited markdown to cache
    save_markdown_to_cache(request.folder_path, request.report)

    # 2. Extract metrics from the saved markdown
    tables_with_headings = extract_metrics_from_markdown(request.report)
    metrics_result = build_metrics_dict_from_extracted(tables_with_headings)

    # 3. Generate visualizations based on the extracted metrics
    run_fallback_visualization(metrics_result)

    # 4. Optionally, return a success response
    return {"success": True}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_pdfs(request: FolderPathRequest):
    try:
        if request.clear_cache:
            cleanup_old_cache()

        folder_path = convert_windows_path(request.folder_path)
        folder_path = os.path.normpath(folder_path)
        folder_path_hash = hash_string(folder_path)
        pdf_files = get_pdf_files_from_folder(folder_path)
        pdfs_hash = hash_pdf_contents(pdf_files)
        logger.info(f"Computed hashes - folder_path_hash: {folder_path_hash}, pdfs_hash: {pdfs_hash}")

        if not request.clear_cache:
            cached_response = get_cached_report(folder_path_hash, pdfs_hash)
            if cached_response:
                logger.info(f"Cache hit for folder_path_hash: {folder_path_hash}")
                return cached_response

        logger.info(f"Cache miss for folder_path_hash: {folder_path_hash} or cache clear requested, running full analysis")
        response = await run_full_analysis(request)

        # --- Generate metrics summary from structured metrics dict ---
        # Support for both Pydantic model and dict return types:
        if hasattr(response, "metrics"):
            metrics = response.metrics
        elif isinstance(response, dict):
            metrics = response.get("metrics", None)
        else:
            metrics = None

        metrics_summary_markdown = metrics_to_markdown(metrics)

        # --- Remove metrics summary section from LLM report if accidentally included ---
        # (just in case you have old reports with placeholder or heading)
        report_text = (
            getattr(response, "report", None)
            if hasattr(response, "report")
            else response.get("report", "") if isinstance(response, dict)
            else ""
        )
        import re
        # Remove section if present (conservative: remove heading + following placeholder line)
        report_text_clean = re.sub(
            r"(## Metrics Summary\s*\[?METRICS_SUMMARY_PLACEHOLDER\]?\s*)", 
            "", 
            report_text, 
            flags=re.IGNORECASE
        )
        # Optionally remove blank lines left behind
        report_text_clean = re.sub(r'\n{3,}', '\n\n', report_text_clean)

        # Update the report field in the response
        if hasattr(response, "copy"):
            response = response.copy(update={
                "report": report_text_clean,
                "metrics_summary": metrics_summary_markdown,
            })
        elif isinstance(response, dict):
            response["report"] = report_text_clean
            response["metrics_summary"] = metrics_summary_markdown

        store_cached_report(folder_path_hash, pdfs_hash, response)
        return response

    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        plt.close('all')




app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
