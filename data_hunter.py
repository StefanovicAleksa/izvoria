# Path: app/data_hunter.py
from sqlalchemy import text, create_engine
from app.core.config import settings


def find_the_data():
    print("========== ONYX DATA HUNTER ==========")
    # Connect to the Postgres system DB to list all other DBs
    base_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/postgres"

    try:
        engine = create_engine(base_url)
        with engine.connect() as conn:
            dbs = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false;")).fetchall()

            print(f"Scanning server at {settings.POSTGRES_SERVER}...")
            for db_row in dbs:
                db_name = db_row[0]
                if db_name == 'postgres': continue

                # Check this specific DB for the vector_chunks table
                try:
                    db_url = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{db_name}"
                    test_engine = create_engine(db_url)
                    with test_engine.connect() as t_conn:
                        # Check if chunks exist
                        count = t_conn.execute(text("SELECT count(*) FROM vector_chunks")).scalar()
                        print(f" -> DB '{db_name}': {count} chunks {' <--- DATA IS HERE!' if count > 0 else ''}")
                except Exception:
                    print(f" -> DB '{db_name}': Table 'vector_chunks' does not exist.")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")


if __name__ == "__main__":
    find_the_data()