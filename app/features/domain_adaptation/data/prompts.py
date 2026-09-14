# Canvas: domain_adaptation_prompts_v2
# Path: app/features/domain_adaptation/data/prompts.py

# Whisper does not follow instructions; its prompt only conditions style and vocabulary on text that
# looks like preceding transcript. A plain sentence that naturally contains the domain terms biases
# spelling toward them without instruction text that can be regurgitated into the transcript.
INITIAL_PROMPT_TEMPLATE = "This recording discusses {domain_names}, including terms such as {terms}."
INITIAL_PROMPT_NO_TERMS_TEMPLATE = "This recording discusses {domain_names}."
