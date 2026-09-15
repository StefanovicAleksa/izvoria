# Canvas: domain_adaptation_prompts_v3
# Path: app/features/domain_adaptation/data/prompts.py

# No Whisper initial prompt is built any more. faster-whisper feeds `initial_prompt` (and `hotwords`) to the model
# as "previous transcript" text, and the sentence prompt used since review batch 3 (9bc403a),
# "This recording discusses {domains}, including terms such as {terms}.", was copied into transcripts as if it had
# been spoken. GlossaryBuilder returns an empty initial prompt; the hotword list carries the vocabulary bias, and
# transcription removes any hotword text Whisper repeats (transcription/utils/prompt_echo_filter.py).

# Router: the model rates every sub-domain instead of listing "the sub-domains". Asked for a list, Llama 3.2 3B
# copied every label in order (all 8 for an ECB/BoE video), so nothing was ever narrowed down.
ROUTER_SYSTEM_PROMPT_TEMPLATE = (
    "You rate how much of a transcript is about each topic.\n"
    "Topics, with typical vocabulary:\n{topic_descriptions}\n\n"
    "Give every topic a score from 0 (not discussed at all) to 10 (a main subject of the transcript), based only "
    "on what the transcript actually says. Output a JSON object mapping each topic to its score."
)
