# domain_adaptation

`app/features/domain_adaptation/`

## What it does

A two-stage "Scout then Route" pipeline that previews a media file's content to auto-detect its industry sub-domain (e.g. a specific legal or medical specialty) and builds a glossary of hotwords/prompt text to bias the main Whisper transcription toward correct domain terminology before the expensive full transcription pass even runs.

## How it does it

`DomainAdaptationAPI.process_domain_adaptation` first uses `SmartSampler.calculate_optimal_slices` to build a representative "trailer" of the audio from the `VadMap` — sampling Start (<33%), Middle (~50%), End (>66%) zones, capped at 120s of speech per zone, to avoid overflowing the LLM context window (or uses everything if the file is short, ≤360s total).

- **Stage 1 ("Scout")**: `ScoutWhisperAdapter` runs a small faster-whisper model (`beam_size=1`, no timestamps) over just those slices to get a rough/dirty preview transcript (`ScoutResult`).
- **Stage 2 ("Router")**: `LlamaAdapter` runs a quantized Llama 3.2 3B via `llama_cpp`, constrained by a GBNF grammar (`JSON_ARRAY_GRAMMAR`) to force strictly-valid JSON array output, classifying the scout text against candidate sub-domain labels loaded from a JSON glossary file (`GlossaryBuilder.get_candidate_labels`).

`GlossaryBuilder.build_injection_payloads` then builds a Whisper `hotword_string` (format `"term:15.0, term:15.0..."`, capped at 40 anchor terms) and an `initial_prompt_string`. Both GPU models are acquired/released individually per stage (never held simultaneously) via `ModelManagerOrchestrator`.

## Inputs / Outputs

Input: `source_id`, `audio_path`, `VadMap`. Output: `DomainGlossary` (`industry`, `sub_domains: List[str]`, `hotword_string`, `initial_prompt_string`) — consumed by `transcription`'s hotwords/prompt parameters.

## Where it fits

Phase 4.5 (`DomainAdaptationPhase`) — runs after VAD, before the main transcription pass, specifically to condition it.

## Gotchas / design notes

- Llama JSON parse failures degrade gracefully (returns `[]` rather than crashing).
- Glossary lookup falls back to base terms if predicted sub-domains have no matching anchors, or to an empty glossary if `industry == "GENERAL"` and no glossary file exists.
