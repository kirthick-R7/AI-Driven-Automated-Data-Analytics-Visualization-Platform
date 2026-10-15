# AI-Driven Automated Data Analytics & Visualization Platform

An **automated data visualization engine** powered by **Google's Gemini AI**.  
It analyzes datasets, generates visualization code, and executes interactive dashboards with built-in **error recovery** and **debugging**.

---

## ✨ Key Features

- 📊 **Automated Visualization Generation** – Transforms raw datasets into interactive dashboards
- 🤖 **AI-Powered Code Generation** – Uses Gemini to create Python visualization code
- 🛡 **Self-Healing Capabilities** – Automatic error detection and correction
- ⚙ **Multiple Backend Support** – Works with standard Python or Streamlit
- 📝 **Comprehensive Logging** – Detailed execution logs for debugging

---

## 🏗 Architecture

### **Core Engine**
- Argument parsing and configuration
- Main execution flow control
- Error handling and recovery

### **Gemini Interface**
- API communication wrapper
- Response processing
- Code extraction and sanitization

### **Execution System**
- Code generation and file management
- Subprocess execution
- Dependency management

### **Debugging System**
- Error analysis and classification
- Automatic dependency installation
- AI-assisted code correction

## 🖼 Project Preview

<p align="center">
  <img width="1986" height="3269" alt="deepseek_mermaid_20250812_c6b8f8" src="https://github.com/user-attachments/assets/c480e646-90af-4bd0-acbf-29f0d37b662d" />
</p>

## ⚙️ Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
2. **Set your Gemini API key**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
## 🚀 Execution
-Run the Auto-Pilot Data Visualization Engine:
  ```bash
  python autoviz.py --dataset your_data.csv
