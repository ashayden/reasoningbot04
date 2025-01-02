# M.A.R.A. - Multi-Agent Reasoning Assistant

## Overview
MARA is an advanced research synthesis tool powered by Google's Gemini Pro model. It employs multiple specialized AI agents to conduct comprehensive research analysis and synthesis on any given topic.

## Key Features

### 1. Quick Insights Generation
- Dynamic "Did You Know?" facts with creative, unexpected connections
- ELI5 (Explain Like I'm 5) explanations using engaging analogies and examples
- Emoji-enhanced visualization for better understanding

### 2. Interactive Focus Areas
- Smart generation of 8-12 relevant focus areas
- Multi-select capability for customized analysis
- Collapsible interface for better organization

### 3. Research Framework Development
- Structured analysis framework with clear objectives
- Methodological approach planning
- Ethical considerations and evaluation criteria

### 4. Multi-Iteration Analysis
- Up to 5 analysis iterations for deeper insights
- Dynamic temperature scaling for creative exploration
- Cross-referencing between iterations
- Comprehensive citation management

### 5. Final Report Generation
- Topic-specific titles and subtitles
- Executive summary with key findings
- Structured analysis sections
- Standardized Works Cited formatting
- Recommended readings section

## Technical Features
- Rate-limited API calls for stability
- Error handling and recovery
- Session state management
- Dynamic markdown formatting
- Standardized citation formatting
- Dark theme UI optimization

## Usage
1. Enter your research topic or question
2. Select number of analysis iterations (1-5)
3. Choose relevant focus areas (optional)
4. Review generated insights and framework
5. Examine each analysis iteration
6. Study the final synthesized report

## Requirements
- Python 3.8+
- Streamlit
- Google Generative AI API access
- Additional dependencies in requirements.txt

## Version History
### v1.3 (Current)
- Enhanced creativity in insights generation
- Standardized Works Cited formatting
- Improved title and subtitle handling
- Better focus area selection interface
- Cleaner markdown formatting

### v1.2
- Added multi-iteration analysis
- Improved error handling
- Enhanced state management
- Better UI responsiveness

### v1.1
- Initial framework implementation
- Basic analysis capabilities
- Simple report generation

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your Google API key in `.streamlit/secrets.toml`:
   ```toml
   GOOGLE_API_KEY = "your-api-key"
   ```
4. Run the application: `streamlit run main.py`

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License
[Insert appropriate license information] 