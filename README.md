# Multi-Agent Reasoning Assistant (MARA)

## Overview
MARA is a sophisticated research analysis system that leverages multiple specialized AI agents to conduct in-depth analysis on any given topic. The system employs a Nobel laureate-level research approach to generate comprehensive, academically rigorous insights while maintaining accessibility.

## Features
- **Pre-Analysis Insights**: Quick, accessible overview with interesting facts and simplified explanations
- **Dynamic Focus Areas**: AI-generated research directions with diverse perspectives
- **Advanced Research Framework**: Sophisticated research structure incorporating multiple theoretical perspectives
- **Iterative Analysis**: Deep, multi-layered research that builds upon previous findings
- **Comprehensive Synthesis**: Detailed final report with executive summary and academic citations

## Agents
1. **PreAnalysisAgent**: Generates initial insights and simplified explanations
2. **PromptDesigner**: Creates focused research directions and optimizes analysis approach
3. **FrameworkEngineer**: Develops comprehensive research frameworks
4. **ResearchAnalyst**: Conducts deep, iterative analysis with Nobel laureate-level expertise
5. **SynthesisExpert**: Synthesizes findings into a cohesive final report

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mara.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Create a .env file with your Google API key:
GOOGLE_API_KEY=your_api_key_here
```

## Usage
1. Run the Streamlit app:
```bash
streamlit run main.py
```

2. Enter your research topic and select the number of analysis iterations (1-5)
3. Review initial insights and select focus areas
4. Monitor the analysis progress
5. Receive a comprehensive final report

## Configuration
- Adjust model parameters in `config.py`
- Customize agent behaviors in `agents.py`
- Modify UI elements in `main.py`

## Technical Details
- **Framework**: Streamlit
- **AI Model**: Google Gemini Pro
- **Python Version**: 3.9+
- **Key Dependencies**:
  - streamlit==1.41.1
  - google-generativeai==0.3.2
  - python-dotenv==1.0.1

## Project Structure
```
mara/
├── main.py           # Main application file
├── agents.py         # Agent implementations
├── config.py         # Configuration settings
├── utils.py          # Utility functions
├── requirements.txt  # Dependencies
└── .streamlit/      # Streamlit configuration
```

## Features in Development
- Enhanced cross-disciplinary analysis
- Advanced visualization of research connections
- Interactive research path selection
- Custom research framework templates

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
MIT License - See LICENSE file for details 