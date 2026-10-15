"""
(Refined: safer API key handling, CLI dataset, improved debugging, logging, retry limits)
"""

import time
import os
import argparse
import subprocess
import sys
import re
import json
from datetime import datetime
import pandas as pd
import google.generativeai as genai
import numpy as np

# ------------------------- CONFIG / DEFAULTS -------------------------
DEFAULT_DATASET = 'SuperStoreOrders.csv'
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# CLI args
parser = argparse.ArgumentParser(description="Auto-pilot data visualization engine (Gemini-powered)")
parser.add_argument("--dataset", default=os.getenv("DATASET", DEFAULT_DATASET), help="Path to CSV dataset")
parser.add_argument("--max-retries", type=int, default=5, help="Maximum debug/retry attempts for generated code")
parser.add_argument("--streamlit", action="store_true", help="If set, attempt to run generated file with 'streamlit run' instead of 'python'")
args = parser.parse_args()

data_set = args.dataset
MAX_RETRIES = args.max_retries
USE_STREAMLIT = args.streamlit

# ------------------------ GEMINI SETUP (secure) -----------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit("Error: GEMINI_API_KEY environment variable not set. Export it before running.")

# GEMINI_API_KEY = "YOUR_API"

genai.configure(api_key=GEMINI_API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

# ------------------------ UTILITIES / HELPERS -------------------------

def timestamp():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def save_log(name, content):
    path = os.path.join(LOG_DIR, f"{timestamp()}_{name}")
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(content, (dict, list)):
            json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            f.write(str(content))
    return path

def safe_send_message(text, retries=3, backoff=2):
    """Send message to Gemini with a small retry/backoff and return text response or raise."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = chat_session.send_message(text)
            # resp may be an object; prefer .text if present
            response_text = getattr(resp, "text", resp)
            return response_text
        except Exception as e:
            last_exc = e
            wait = backoff * attempt
            print(f"[Gemini] send_message failed (attempt {attempt}/{retries}), retrying in {wait}s...")
            time.sleep(wait)
    # final failure
    raise RuntimeError(f"Failed to send message to Gemini after {retries} attempts") from last_exc

def gen_code(data_snip):
    """Ask Gemini to analyze dataset snippet and produce Streamlit dashboard code."""
    # First a simple analysis prompt (we keep your original behavior)
    try:
        safe_send_message("Analyse the given data set " + data_set + " : " + data_snip)
    except Exception as e:
        print("[Warning] initial analysis prompt failed:", e)

    prompt = "generate a python code to create a dashboard with all the possible visualization for the data set " + data_set + " on streamlit"
    response = safe_send_message(prompt)
    # Save raw response for audit
    resp_path = save_log("gemini_response.txt", response)
    print(f"[Log] Gemini response saved to: {resp_path}")
    return response

def trim(response):
    """Safely extract a python code block from Gemini response or fall back to whole text."""
    if not isinstance(response, str):
        response = str(response)
    if "```python" in response:
        try:
            code = response.split("```python", 1)[1].split("```", 1)[0]
        except Exception:
            code = response
    elif "```" in response and "python" in response.lower():
        try:
            # handle variations like ```py
            code = response.split("```", 1)[1].split("```", 1)[0]
        except Exception:
            code = response
    else:
        # fallback: attempt to isolate lines that look like code (heuristic)
        lines = response.splitlines()
        code_lines = [ln for ln in lines if ln.strip().startswith(("import ", "from ", "def ", "class ", "if ", "#")) or "pd." in ln or "streamlit" in ln]
        code = "\n".join(code_lines) if code_lines else response
    return code.strip()

def str_to_py(code_string, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(code_string)

def py_to_str(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def install_missing_modules_from_error(error_text):
    """
    Identify missing modules from error text and install them.
    Handles patterns like: ModuleNotFoundError: No module named 'xyz'
    """
    modules = set(re.findall(r"(?:No module named '([^']+)'|ModuleNotFoundError: No module named '([^']+)')", error_text))
    # re.findall returns tuples when alternation used; flatten
    flat = set()
    for tup in modules:
        for name in tup:
            if name:
                flat.add(name)
    # Also try another pattern (pip package naming may differ from module name; we use module as-is)
    if not flat:
        alt = re.findall(r"^\s*ImportError: cannot import name '([^']+)'", error_text, flags=re.IGNORECASE | re.MULTILINE)
        flat.update(alt)
    if not flat:
        return []
    installed = []
    for module in flat:
        try:
            print(f"[Installer] Installing missing module: {module}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', module])
            installed.append(module)
        except subprocess.CalledProcessError as e:
            print(f"[Installer] pip failed for {module}: {e}")
    return installed

def extract_error_info(traceback_text):
    """Extract error type, missing key or name, and line number from a traceback string."""
    error_type_match = re.search(r'(\w+Error):', traceback_text)
    # KeyError: 'something'
    missing_key_match = re.search(r"KeyError: '([^']+)'", traceback_text)
    # ModuleNotFoundError: No module named 'pandas'
    module_match = re.search(r"No module named '([^']+)'", traceback_text)
    line_number_match = re.search(r'File ".+?", line (\d+)', traceback_text)
    return {
        'error_type': error_type_match.group(1) if error_type_match else "UnknownError",
        'missing_key': missing_key_match.group(1) if missing_key_match else (module_match.group(1) if module_match else "UnknownKey"),
        'line_number': line_number_match.group(1) if line_number_match else "UnknownLine",
        'traceback': traceback_text
    }

def debug(filename, code, error, key_error, complex_flag):
    """
    Attempt to fix code by:
    - installing missing module(s) if error indicates that
    - otherwise asking Gemini to provide a corrected full code
    """
    tb = error
    if "No module named" in tb or "ModuleNotFoundError" in tb:
        installed = install_missing_modules_from_error(tb)
        if installed:
            print(f"[Debug] Installed modules: {installed}")
            # after installing, we return to the run loop (which will re-run the file)
            return
    else:
        # Ask Gemini to regenerate the full code. Keep the prompt simple and explicit.
        # We include the original code and the error to help Gemini.
        query = (
            "The following Python code produced this error:\n\n"
            f"Error:\n{tb}\n\n"
            f"Code:\n{code}\n\n"
            "Please provide the full corrected Python code. Always return the code inside a ```python block. "
            "If you cannot fully fix it, include a final comment explaining what to change."
        )
        try:
            response = safe_send_message(query, retries=4, backoff=3)
        except Exception as e:
            print("[Debug] Gemini failed to provide a fix:", e)
            return
        # Save Gemini's debug response for audit
        save_log("gemini_debug_response.txt", response)
        corrected = trim(response)
        str_to_py(corrected, filename)
        print("[Debug] Rewrote generated file with Gemini's corrected code.")

# -------------------------- RUN / DEBUG LOOP --------------------------

def run(file):
    """
    Run the generated python file (or streamlit run if requested). On failure,
    parse error, try to auto-install packages, or ask Gemini to regenerate code.
    """
    error_history = []
    attempt = 0

    while True:
        attempt += 1
        if attempt > MAX_RETRIES:
            print(f"[Run] Reached max retries ({MAX_RETRIES}). Aborting.")
            break

        try:
            if USE_STREAMLIT:
                # run Streamlit in a way similar to how user would run it; we wait for process to exit
                cmd = ['streamlit', 'run', file]
            else:
                cmd = ['python', file]

            print(f"[Run] Executing ({' '.join(cmd)}) (attempt {attempt})...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("[Run] Output:\n", result.stdout)
            # On successful run, break out of loop
            break

        except subprocess.CalledProcessError as e:
            stderr = e.stderr or e.output or str(e)
            info = extract_error_info(stderr)
            error_history.append(info)
            print(f"[Run] Error (attempt {attempt}): {info['error_type']} at line {info['line_number']}")
            # Save traceback for audit
            save_log(f"traceback_attempt_{attempt}.txt", stderr)
            code = py_to_str(file)
            print("--- Generated code snapshot (for debug) ---")
            print(code)
            print("------------------------------------------")
            # If same error repeated twice in a row, escalate (complex_flag=1)
            complex_flag = 0
            if len(error_history) >= 2 and error_history[-1] == error_history[-2]:
                complex_flag = 1
            debug(file, code, stderr, info, complex_flag)
            # small pause to avoid tight loop (but keep short)
            time.sleep(2)
        except Exception as e:
            # non CalledProcessError exceptions
            tb = str(e)
            save_log("unexpected_error.txt", tb)
            print("[Run] Unexpected error:", tb)
            break

# -------------------------- MASTER FLOW --------------------------

def master(data_set):
    df = pd.read_csv(data_set)
    columns = list(df.columns)
    print("COLUMN : \n", columns, "\n\n")
    global data
    data = {}
    for col in columns:
        data[col] = list(df[col].head(10))

    print("DATA SET SNIPPET : \n", data, "\n\n")
    global data_snip
    data_snip = str(data)

    # Ask Gemini for code
    response = gen_code(data_snip)
    print("RESPONSE FROM GEMINI SAVED. Extracting code...")
    code = trim(response)
    file = 'generated.py'
    str_to_py(code, file)
    print(f"[Master] Generated code written to {file}")
    run(file)

# ------------------------------- START -------------------------------
if __name__ == "__main__":
    master(data_set)
