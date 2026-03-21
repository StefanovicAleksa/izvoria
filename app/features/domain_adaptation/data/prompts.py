# Path: app/features/domain_adaptation/data/prompts.py

INITIAL_PROMPT_TEMPLATE = (
    "The following dictation covers topics related to {domain_names}. "
    "It contains specialized terminology such as: {anchor_words}."
)