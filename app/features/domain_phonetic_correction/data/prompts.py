# Path: app/features/domain_phonetic_correction/data/prompts.py

# The model never writes text. For each candidate it only picks which of two versions of the sentence the speaker
# said: as transcribed, or with the dictionary term. Asked as a batch of true/false verdicts, the 3B model rejected
# every real mis-hearing; a single A/B choice, asked in both orders, separated them without a false acceptance.
PHONETIC_VERIFY_SYSTEM_PROMPT = """You compare two versions of one sentence from a speech-recognition transcript of a recording about {domain}.
The versions differ only in the words in [brackets]: one keeps what the recognizer wrote, the other replaces it with a dictionary term that sounds similar.
Speech recognition often mishears a rare domain term as common words that sound alike, but ordinary words used normally are usually transcribed correctly.
Pick the version the speaker most likely said. Answer with the letter only."""

PHONETIC_VERIFY_USER_TEMPLATE = """Previous sentence: {previous_text}

A: {version_a}
B: {version_b}

Which version did the speaker say?"""
