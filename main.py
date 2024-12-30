import logging
from core import initialize_gemini, analyze_topic

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Command-line interface for the Gemini Reasoning Bot."""
    print("\n=== Gemini Reasoning Bot ===\n")
    
    # Initialize Gemini
    if not initialize_gemini():
        print("Error: Failed to initialize Gemini API. Please check your API key.")
        return
    
    try:
        # Get user input
        print("Agent 1: Let me gather some information from you.")
        topic = input("What topic should we explore? ")
        loops = int(input("How many reasoning iterations? "))
        
        # Perform analysis
        print("\nStarting analysis...")
        result = analyze_topic(topic, loops)
        
        # Display results
        print("\n=== Analysis Framework ===")
        print(result['framework'])
        
        print("\n=== Detailed Analysis ===")
        for i, analysis in enumerate(result['analysis'], 1):
            print(f"\nIteration {i}:")
            print(analysis)
            print("-" * 50)
        
        print("\n=== Final Report ===")
        print(result['summary'])
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Please try again. If the error persists, try fewer iterations.")

if __name__ == "__main__":
    main()

