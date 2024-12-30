import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Agent 1: Creates sophisticated prompt structure (low temperature for focused, consistent output)
print("Agent 1: Let me gather some information from you.")
topic = input("What topic should we explore? ")
loops = int(input("How many reasoning iterations? "))

print("\nAgent 1: Formulating analysis framework...")
prompt_response = model.generate_content(
    f"""You are an expert prompt engineer. Your task is to take the '{topic}' and create a refined system prompt for a reasoning agent and a structured framework for analysis.
    Include instructions for examining multiple perspectives, potential implications, and interconnected aspects.
    Be specific but concise.""",
    generation_config=genai.types.GenerationConfig(
        temperature=0.3
    ),
    stream=True
)

system_prompt = ""
for chunk in prompt_response:
    system_prompt += chunk.text
    print(chunk.text, end='', flush=True)

# Agent 2: Reasoning agent (high temperature for creative insights)
full_analysis = []
context = topic
for i in range(loops):
    print(f"\n\nReasoning Loop {i+1}/{loops}:")
    response = model.generate_content(
        f"""{system_prompt}
        
        Previous context: {context}
        Analyze this topic as if you were a Nobel Prize winner in the relevant field, drawing upon deep expertise and groundbreaking insights. Provide fresh analysis following the framework above.""",
        generation_config=genai.types.GenerationConfig(
            temperature=1.0
        ),
        stream=True
    )
    context = ""
    loop_content = ""
    for chunk in response:
        context += chunk.text
        loop_content += chunk.text
        print(chunk.text, end='', flush=True)
    full_analysis.append(loop_content)
    print("\n" + "-"*50)

# Agent 3: Summarization agent (very low temperature for consistent, focused summary)
print("\n\nAgent 3: Final Summary")
summary = model.generate_content(
    f"""You are an expert analyst. Provide a comprehensive final report with the following structure: 
    Synthesize the findings from the analysis loops about '{topic}' into a Final Report with the following structure: 

    1. Executive Summary
- A concise overview of the investigation and key findings (2-3 paragraphs)

2. Key Insights
- Bullet-pointed list of the most important discoveries and conclusions
- Focus on actionable and noteworthy findings
- Include surprising or counter-intuitive insights

3. Analysis
[Scale analysis depth based on research loops]
- Synthesize major concepts and themes from the research
- Examine relationships between different aspects
- Support claims with evidence from the research
- Address any contradictions or nuances found

4. Supplementary Synthesis
[Dynamic section based on topic and research depth]
Choose relevant elements from:
- Recommendations for action or further investigation
- Implications of the findings
- Counter-arguments or alternative perspectives
- Significance and broader impact
- Limitations of current understanding
- Future trends or developments

5. Conclusion
- Summarize the most important takeaways
- Place findings in broader context
- Highlight remaining questions or areas for future research

6. Further Learning
- List key sources referenced in the analysis
- Recommend additional reading materials
- Finally, suggest 3 follow-up questions:
    - 1 that digs deeper into a key aspect of the topic
    - 1 that explores a related topic
    - 1 that investigates surprising orunexpected connections to the topic:

Write in a clear, authoritative tone. Support all major claims with evidence from the research
    

    {' '.join(full_analysis)}""",
    generation_config=genai.types.GenerationConfig(
        temperature=0.1
    ),
    stream=True
)

for chunk in summary:
    print(chunk.text, end='', flush=True)

