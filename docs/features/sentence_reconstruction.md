# sentence_reconstruction

`app/features/sentence_reconstruction/`

## What it does

Reconstructs fragmented, unpunctuated acoustic transcript text into readable, properly-bounded sentences — turning Whisper's raw word stream plus diarization's speaker labels into clean sentence-level units for the rest of the pipeline.

## How it does it

**Which adapter** depends on the language pack's `punctuation.source`, chosen in `bootstrap.py`. The fullstop classifier was trained only on English, German, French and Italian. For other languages (`sr`, `sl`), `WhisperPunctuationAdapter` tags each word with the mark Whisper itself wrote after it (`requires_model = False`): `SentenceReconstructionAPI` then acquires nothing, and the phase queues no preload.

`PunctuationAdapter` loads `fullstop-punctuation-multilingual-base` with `AutoTokenizer` + `AutoModelForTokenClassification` (GPU when available). It feeds the acoustic words pre-split (`is_split_into_words=True`) in overlapping windows of 230 words (5-word overlap; a window is shrunk rather than truncated if it exceeds 512 tokens) and maps predictions back with `word_ids()`: each word takes the label of its **last** subtoken, where the model places the punctuation mark (as its reference library does). Whisper's own punctuation is removed from each word before inference (`(?<!\d)[.,;:!?](?!\d)`, which keeps decimal points), as the reference library does. Left in, the last subtoken of "develop," is the comma itself, labelled "no punctuation", and the mark was erased. Later windows re-label overlap words, since they see right-hand context. Labels map exactly 1:1 to words; the model's `"0"` (no punctuation) and any unknown label become `NONE`. Critically, the model **only classifies** punctuation per word — it never generates text — which is a deliberate anti-hallucination design choice consistent with the project's zero-fabrication requirement.

`SentenceReconstructionAPI.reconstruct_transcript` groups words by speaker first and **never merges text across a speaker wall** (a sentence never spans two different speakers), tags punctuation per word via the adapter, then performs pure-Python sentence slicing (`deterministic_assembler.py`) based on those punctuation tags — uppercasing only the first letter of a sentence (never lowercasing acronyms such as "MRI") and rendering only real punctuation marks, never a tag as literal text — and joining words the way Whisper spaced them: a word without a leading space is a continuation ("160" + ".99", "pre" + "-planning") that attaches to the previous word, and the word before it never takes punctuation or ends a sentence — and persists the result via `ISentenceRepository`. It requires diarization to have already run, raising `MissingDiarizationError` otherwise (since speaker boundaries are load-bearing for correct sentence grouping).

## Inputs / Outputs

Input: a diarized `Transcript`. Output: a `ReconstructedTranscript` (`sentences: List[ReconstructedSentence]`), each sentence carrying its speaker id and timing.

## Where it fits

Phase 7 (`SentenceReconstructionPhase`) — runs after diarization, before domain phonetic correction (which operates per-sentence).

## Gotchas / design notes

- Overlapping word windows (instead of truncation) are what let arbitrarily long speaker turns be punctuated completely.
- The earlier HF pipeline + character-offset mapping took the first overlapping entity group, which dropped punctuation on multi-subtoken words and could leave a literal `"0"` in the text; the unit tests in `tests/features/sentence_reconstruction/` pin the last-subtoken behaviour.
