# sentence_reconstruction

`app/features/sentence_reconstruction/`

## What it does

Reconstructs fragmented, unpunctuated acoustic transcript text into readable, properly-bounded sentences — turning Whisper's raw word stream plus diarization's speaker labels into clean sentence-level units for the rest of the pipeline.

## How it does it

`PunctuationAdapter` runs a Hugging Face `token-classification` pipeline (`fullstop-punctuation-multilingual-base`), GPU-bound, chunked into 512-token windows with a 16-token stride to avoid truncating long transcripts. Critically, the model **only classifies** punctuation per word — it never generates text — which is a deliberate anti-hallucination design choice consistent with the project's zero-fabrication requirement.

`SentenceReconstructionAPI.reconstruct_transcript` groups words by speaker first and **never merges text across a speaker wall** (a sentence never spans two different speakers), tags punctuation per word via the adapter, then performs pure-Python sentence slicing (`deterministic_assembler.py`) based on those punctuation tags, and persists the result via `ISentenceRepository`. It requires diarization to have already run, raising `MissingDiarizationError` otherwise (since speaker boundaries are load-bearing for correct sentence grouping).

## Inputs / Outputs

Input: a diarized `Transcript`. Output: a `ReconstructedTranscript` (`sentences: List[ReconstructedSentence]`), each sentence carrying its speaker id and timing.

## Where it fits

Phase 7 (`SentenceReconstructionPhase`) — runs after diarization, before domain phonetic correction (which operates per-sentence).

## Gotchas / design notes

The 512-token/16-token-stride windowing exists specifically so very long transcripts don't get silently truncated by the underlying HF pipeline — a subtle correctness bug this project guards against explicitly (see the mirrored unit test in `tests/features/sentence_reconstruction/`).
