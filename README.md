# M.A.R.A. - Multi-Agent Reasoning Assistant

A sophisticated multi-agent reasoning system powered by Google's Gemini Pro 1.5 model. This application employs four specialized agents to perform comprehensive analysis on any given topic:

0. **Prompt Designer** (✍️): Creates optimized prompts for precise analysis
1. **Framework Engineer** (🎯): Develops structured research frameworks
2. **Research Analyst** (🔄): Conducts iterative, in-depth analysis
3. **Synthesis Expert** (📊): Generates comprehensive final reports

## Features

- **Multi-Agent Architecture**: Four specialized agents working in concert
- **Configurable Analysis Depth**: Choose from Quick (1 iteration) to Comprehensive (4 iterations)
- **Dynamic Research Framework**: Tailored framework generation for each topic
- **Progressive Analysis**: Iterative research that builds upon previous findings
- **Comprehensive Reporting**: Detailed final reports with executive summaries and key insights
- **Modern UI**: Clean, responsive interface with collapsible sections

## Requirements

- Python 3.8+
- Streamlit account (for deployment)
- Google AI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ashayden/gemini-reasoning-bot.git
cd gemini-reasoning-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Streamlit secrets:
   - Create `.streamlit/secrets.toml`
   - Add your Google AI API key:
     ```toml
     GOOGLE_API_KEY = "your_api_key_here"
     ```

## Usage

1. Run the application locally:
```bash
streamlit run main.py
```

2. Or deploy to Streamlit Cloud:
   - Push to your GitHub repository
   - Connect repository to Streamlit Cloud
   - Add your API key in Streamlit Cloud secrets

## Using the Application

1. Enter your topic of interest in the text input
2. Select your desired analysis depth:
   - Quick: Single-pass analysis
   - Balanced: Two iterations
   - Deep: Three iterations
   - Comprehensive: Four iterations
3. Click "Start Analysis" to begin
4. Watch as each agent performs its specialized task:
   - Prompt optimization
   - Framework creation
   - Iterative research analysis
   - Final synthesis and reporting

## Security Note

Never commit API keys or secrets to the repository. Use Streamlit's secrets management system for secure credential handling. 