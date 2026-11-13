# Intelligent Data Analysis Agent

An AI-powered assistant for automated Exploratory Data Analysis (EDA). This project combines Python-based analytical tools with a locally running Large Language Model (LLM) to help users explore datasets through a simple interface and natural language questions.

## Features
- Upload and explore CSV datasets.
- View dataset information and data quality.
- Perform descriptive statistical analysis.
- Analyze correlations and data distributions.
- Generate data visualizations.
- Ask questions using the AI Data Assistant.
- Generate and download analysis reports.

## System Architecture

The application follows a single-agent architecture that combines rule-based tool selection, Python-based data analysis and a locally running LLM.

**Workflow:**

1. **User Input:** Uploads a CSV dataset or asks a question.
2. **Dataset Processing:** Pandas loads and processes the dataset.
3. **Tool Selection:** The `select_tools()` function identifies suitable analytical tools based on keywords in the user's question.
4. **Tool Execution:** The `execute_tools()` function runs the selected Python analytical functions.
5. **AI Response:** The `ask_ai()` function uses direct Pandas calculations or Ollama with Llama 3.2 3B to generate responses.
6. **Output:** Displays analysis results, visualizations and downloadable reports.

<img width="921" height="633" alt="image" src="https://github.com/user-attachments/assets/6204e691-e54f-404f-bded-984b45c3925d" />

## Technologies Used
- Python
- Streamlit
- Pandas and NumPy
- Matplotlib
- Ollama
- Llama 3.2 3B

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Sonam43/data-analysis.git
   ```

2. Open the project folder in VS Code.

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Make sure Ollama is installed and run the model:

   ```bash
   ollama run llama3.2:3b
   ```

5. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

The application will open in your local browser.

## Application Overview

The following screenshots provide an overview of the Intelligent Data Analysis Agent and its main features.
<img width="1356" height="639" alt="image" src="https://github.com/user-attachments/assets/ca8b57b8-4ece-4c64-bae8-38a780d09a87" />
<img width="1357" height="628" alt="image" src="https://github.com/user-attachments/assets/627340d7-e7e3-48be-9581-b1b46bddafb3" />
<img width="1344" height="596" alt="image" src="https://github.com/user-attachments/assets/6a8efe7d-d779-494a-bbb6-1490576a50c5" />

**Note:** The application currently runs locally using Ollama and has not been deployed on Streamlit Cloud.
