# Canvas: debug_main_v3
# Path: app/debug_main.py

import argparse
import os
from typing import Dict

# CRITICAL FIX: Stop PyTorch memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from app.main import bootstrap_database, get_orchestration_pipeline
from app.features.storage.domain.entities import SourceType
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult
from app.orchestration.exceptions.workflow import WorkflowExecutionError


class DebugMediaPipeline:
    def __init__(self, base_pipeline):
        self._base = base_pipeline

    def process_directory(self, target_directory: str, source_type: SourceType) -> Dict[
        PipelinePhase, PhaseBatchResult]:
        pipeline_results: Dict[PipelinePhase, PhaseBatchResult] = {}
        try:
            scan_res = self._base._scan_phase.execute(target_directory)
            pipeline_results[PipelinePhase.SCAN] = scan_res
            if scan_res.status == PhaseStatus.FAILED or scan_res.successful_count == 0: return pipeline_results

            ingest_res = self._base._ingest_phase.execute(scan_res.items, source_type)
            pipeline_results[PipelinePhase.INGEST] = ingest_res
            video_ids = [i.source_id for i in ingest_res.items if i.success and i.source_id]
            if not video_ids: return pipeline_results

            self._base._vad_phase.preload_model()
            ext_res = self._base._extract_audio.execute(video_ids)
            pipeline_results[PipelinePhase.AUDIO_EXTRACTION] = ext_res
            ext_items = [i for i in ext_res.items if i.success]
            if not ext_items: return pipeline_results

            vad_res = self._base._vad_phase.execute(ext_items)
            pipeline_results[PipelinePhase.VAD] = vad_res
            vad_items = [i for i in vad_res.items if i.success]
            if not vad_items: return pipeline_results

            dom_res = self._base._domain_adaptation_phase.execute(vad_items)
            pipeline_results[PipelinePhase.DOMAIN_ADAPTATION] = dom_res

            self._base._transcribe_phase.preload_model()
            trans_res = self._base._transcribe_phase.execute(vad_items)
            pipeline_results[PipelinePhase.TRANSCRIBE] = trans_res

            print("\n[DEBUG] Pipeline successfully halted after Transcription.")
            return pipeline_results
        except Exception as e:
            raise WorkflowExecutionError(f"Debug Pipeline Crash: {str(e)}", original_exception=e) from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=str)
    args = parser.parse_args()

    print("==================================================")
    print("  ONYX DEBUG PIPELINE (STOP AT TRANSCRIBE)")
    print("==================================================\n")

    bootstrap_database()
    base_pipeline = get_orchestration_pipeline()
    debug_pipeline = DebugMediaPipeline(base_pipeline)

    try:
        results = debug_pipeline.process_directory(args.target_dir, SourceType.VIDEO)
        for phase, res in results.items():
            print(f"\n[{phase.value.upper()}] Status: {res.status.value.upper()} | {res.successful_count} Success")
            if res.errors:
                for err in res.errors: print(f"    - {err}")
    except Exception as e:
        print(f"\n[FATAL] {str(e)}")


if __name__ == "__main__":
    main()