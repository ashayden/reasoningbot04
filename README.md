# Gemini Reasoning Bot

A sophisticated multi-agent reasoning system powered by Google's Gemini Pro 1.5 model. This application leverages three specialized agents to perform deep analysis on any given topic:

1. **Agent 1 (Prompt Engineer)**: Creates a refined system prompt and analysis framework
2. **Agent 2 (Reasoning Agent)**: Performs multiple iterations of deep analysis
3. **Agent 3 (Summarization Agent)**: Synthesizes findings into a comprehensive final report

## Features

- Multi-agent architecture for sophisticated reasoning
- Configurable number of reasoning iterations
- Structured analysis framework
- Comprehensive final report generation
- Real-time streaming of model responses

## Requirements

- Python 3.8+
- Google Generative AI API key

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

3. Set up your environment:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Edit `.env` and replace `your_api_key_here` with your actual API key

## Usage

Run the main script:
```bash
python main.py
```

Follow the prompts to:
1. Enter your topic of interest
2. Specify the number of reasoning iterations
3. Watch as the agents analyze and synthesize information about your topic

## Security Note

The API key is stored in the `.env` file, which is excluded from git via `.gitignore`. Never commit your actual API key to the repository. 