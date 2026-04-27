# Canvas: diagnostic_check_v2
# Path: app/diagnostic_check.py

from sqlalchemy import text
from app.core.database.service.manager import db_manager


def run_diagnostics():
    print("--- Onyx Database Diagnostic ---")
    try:
        with db_manager.session() as session:
            # 1. Check Chunk Count
            count = session.execute(text("SELECT COUNT(*) FROM vector_chunks")).scalar()
            print(f"Total Chunks in vector_chunks: {count}")

            # 2. Check Clip Linkages
            clips = session.execute(text("SELECT COUNT(*) FROM video_clips")).scalar()
            print(f"Total Linked Video Clips: {clips}")

            if count > 0:
                # 3. Check for specific Central Bank data
                print("\nSearching for 'Central Bank' in text_payload...")
                search = session.execute(
                    text("SELECT count(*) FROM vector_chunks WHERE text_payload ILIKE '%central bank%'")
                ).scalar()
                print(f"Chunks containing 'central bank': {search}")

                # 4. Preview Data
                print("\nData Preview:")
                rows = session.execute(
                    text("SELECT text_payload, length(embedding) FROM vector_chunks LIMIT 1")).fetchone()
                if rows:
                    print(f"Sample Text: {rows[0][:100]}...")
                    print(f"Embedding Dimensions: {rows[1]}")
            else:
                print("\n❌ CRITICAL: The search index is EMPTY.")
                print(
                    "Possible reasons: Ingestion failed, commit was rolled back, or you are looking at the wrong database.")
    except Exception as e:
        print(f"Diagnostic failed: {e}")


if __name__ == "__main__":
    run_diagnostics()