# domain_adaptation

`app/features/domain_adaptation/`

## What it does

A two-stage "Scout then Route" pipeline that previews a media file's content to auto-detect its industry sub-domain (e.g. a specific legal or medical specialty) and builds a glossary of hotwords/prompt text to bias the main Whisper transcription toward correct domain terminology before the expensive full transcription pass even runs.

## How it does it

`DomainAdaptationAPI.process_domain_adaptation` first uses `SmartSampler.calculate_optimal_slices` to build a representative "trailer" of the audio from the `VadMap` — sampling Start (<33%), Middle (~50%), End (>66%) zones, capped at 120s of speech per zone, to avoid overflowing the LLM context window (or uses everything if the file is short, ≤360s total).

- **Stage 1 ("Scout")**: `ScoutWhisperAdapter` runs a small faster-whisper model (`beam_size=1`, no timestamps) over just those slices to get a rough/dirty preview transcript (`ScoutResult`).
- **Stage 2 ("Router")**: `LlamaAdapter` runs a quantized Llama 3.2 3B via `llama_cpp`, scoring the scout text against every candidate sub-domain from the JSON glossary (`GlossaryBuilder.get_candidate_descriptions`: label → its first anchor words and terms). A GBNF grammar built per call makes it emit every label exactly once with a 0–10 score, and labels scoring at least `DOMAIN_ROUTER_MIN_SCORE` (5) are used, highest first. Asking for "the sub-domains" as a list made the model copy every label in order.

`GlossaryBuilder.build_injection_payloads` then builds a Whisper `hotword_string` — a plain comma-separated list (`"strike, theta, ..."`, capped at 40 anchor terms); faster-whisper inserts hotwords into the prompt as text and has no weighting syntax — and an always-empty `initial_prompt_string`. faster-whisper feeds hotwords and the initial prompt to the model as "previous transcript" text, and the sentence prompt used before ("This recording discusses …, including terms such as …") was copied into transcripts as if it had been spoken. Transcription also removes any hotword run Whisper repeats. Both GPU models are acquired/released individually per stage (never held simultaneously) via `ModelManagerOrchestrator`.

## Inputs / Outputs

Input: `source_id`, `audio_path`, `VadMap`. Output: `DomainGlossary` (`industry`, `sub_domains: List[str]`, `hotword_string`, `initial_prompt_string`) — consumed by `transcription`'s hotwords/prompt parameters.

## Where it fits

Phase 4.5 (`DomainAdaptationPhase`) — runs after VAD, before the main transcription pass, specifically to condition it.

## Gotchas / design notes

- Llama JSON parse failures degrade gracefully (returns `[]` rather than crashing).
- Glossary lookup falls back to base terms if predicted sub-domains have no matching anchors, or to an empty glossary if `industry == "GENERAL"` and no glossary file exists.
- A transcript that isn't clearly about any candidate sub-domain gets `[]` (every score below the threshold), and hotwords fall back to the base terms.
- The scout hears at most 2 minutes of speech from each of the start, middle and end. A topic discussed only outside those windows scores 0 (an ECB/BoE drill video scored Bank of England 0). Too few sub-domains means fewer hotwords and fewer phonetic corrections, never wrong ones.
