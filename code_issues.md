# Code Issues and Resolutions

## Import Issues
### Issue: Config Import Error in utils.py
- **Problem**: Application failed to find `config` module when importing `from config import config` in `utils.py`
- **Root Cause**: The `config` variable was not defined in `config.py`
- **Resolution**: Updated import statement to import specific configuration settings instead of non-existent `config` variable

## UI/UX Issues
### Issue: Markdown Rendering in Titles
- **Problem**: Literal asterisks ("**") showing up in headings, titles, and subtitles
- **Root Cause**: Markdown formatting not being properly processed
- **Resolution**: Added markdown cleaning function to strip literal markdown characters from titles

### Issue: Button Centering
- **Problem**: Text and emojis in buttons (Start Analysis, Skip, Continue) not centered properly
- **Root Cause**: CSS flexbox properties not properly configured
- **Resolution**: Updated button CSS to use `inline-flex` with proper centering properties

## Content Generation Issues
### Issue: ELI5 Output Format
- **Problem**: ELI5 explanations included meta-references and overly simplified language
- **Root Cause**: Prompt instructions leading to child-like explanations
- **Resolution**: Updated prompt to focus on clear, direct overviews without meta-references

### Issue: Title/Subtitle Formatting
- **Problem**: Research sections showing "Title:" and "Subtitle:" labels in output
- **Root Cause**: Title parsing not removing label prefixes
- **Resolution**: Updated title parsing to remove labels while maintaining hierarchy

## State Management Issues
### Issue: Focus Area Selection State
- **Problem**: Focus area selections not persisting between reruns
- **Root Cause**: Session state not properly handling focus area selections
- **Resolution**: Added dedicated session state variables for focus area tracking

## Deployment Issues
### Issue: Streamlit Cloud Tag Visibility
- **Problem**: Version tag `v1.3.2` not visible in Streamlit Cloud deployment options
- **Root Cause**: Tag not properly pushed to GitHub
- **Resolution**: Used `version-1.3.1-stable` branch for deployment as temporary solution 

## API Issues
### Issue: API Quota Exhaustion (Error 429)
- **Problem**: Application fails with "429 Resource has been exhausted" error during content generation
- **Root Cause**: Google API quota limits being reached
- **Impact**: Affects all content generation stages (insights, analysis, synthesis)
- **Solution Implemented**:
  1. Created `APIRateLimiter` class to manage quota tracking
  2. Implemented smart cooldown periods (5 minutes after multiple 429s)
  3. Added error counting to detect repeated quota issues
  4. Enhanced error handling with user-friendly countdown timer
  5. Added quota reset tracking
- **Future Considerations**:
  1. Implement request caching for frequently accessed content
  2. Add request rate monitoring dashboard
  3. Consider implementing multiple API keys for load balancing 