# Canvas: contextual_prompts_v1
# Path: app/features/contextual_retrieval/data/prompts.py

# This prompt is strictly adapted from Anthropic's Contextual Retrieval research.
# It is designed to be highly directive to prevent Small Language Models (SLMs)
# from hallucinating or answering conversational questions.

CONTEXTUAL_SYSTEM_PROMPT = """You are an elite, highly constrained context-generation AI.
Your sole purpose is to provide short, succinct situational context for a specific chunk of text based on the full document provided.
Do not summarize the entire document. Do not answer questions. Do not use conversational filler (e.g., "Here is the context:").
The document is a transcript: each line is a speaker turn such as "Speaker_0: ...". Never start your answer with a speaker label and never copy lines from the chunk.
Output ONLY the contextual text, written as one or two plain sentences."""

CONTEXTUAL_USER_PROMPT_TEMPLATE = """<document> 
{full_document_text} 
</document> 

Here is the chunk we want to situate within the whole document:
<chunk> 
{chunk_text} 
</chunk> 

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""