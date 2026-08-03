"""
Idempotent, incremental schema changes for existing dev databases.

There is no Alembic in this project. schema.sql only runs once, against a
fresh database (see init_db.py's has_table("users") check) — every table or
column added after that has to reach existing dev DBs some other way. The
convention (already used for the lawyers.bio_embedding pgvector column
before this file existed) is: add the change to schema.sql for fresh
installs, AND add a matching idempotent `ensure_*` function here, called
unconditionally (in dependency order — parents before FK children) from
run_migrations(). Every ensure_* function must be safe to run on every
startup, on a DB that may or may not already have the change applied.
"""

from sqlalchemy.engine import Engine


def ensure_lawyer_embedding_column(engine: Engine) -> None:
    """pgvector column on `lawyers`, used for semantic lawyer matching."""
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.exec_driver_sql(
            "ALTER TABLE lawyers ADD COLUMN IF NOT EXISTS bio_embedding vector(1024);"
        )


def ensure_users_role_column(engine: Engine) -> None:
    """"client" | "lawyer" | "admin" — lets lawyer-scoped features (Vault
    sharing, notifications) distinguish account types without a new table."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'client';"
        )


def ensure_lawyers_user_id_column(engine: Engine) -> None:
    """Nullable FK so a directory listing can optionally be claimed by a
    real login. Depends on ensure_users_role_column having created `users`
    already (it always exists by this point, added defensively anyway)."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE lawyers ADD COLUMN IF NOT EXISTS user_id TEXT "
            "REFERENCES users(id) ON DELETE SET NULL;"
        )


def ensure_similar_case_searches_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS similar_case_searches (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                uploaded_doc_summary TEXT NOT NULL,
                firac JSONB NOT NULL,
                results JSONB NOT NULL,
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def ensure_case_tables(engine: Engine) -> None:
    """My Cases: saved_cases + case_events + case_notes + case_ai_summaries."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS saved_cases (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                cnr TEXT,
                court TEXT,
                case_number TEXT,
                year INTEGER,
                title TEXT,
                status TEXT,
                last_synced_at TIMESTAMPTZ(6),
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS saved_cases_user_id_cnr_key "
            "ON saved_cases(user_id, cnr) WHERE cnr IS NOT NULL;"
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS case_events (
                id TEXT PRIMARY KEY,
                saved_case_id TEXT NOT NULL REFERENCES saved_cases(id) ON DELETE CASCADE,
                event_type TEXT NOT NULL,
                event_date TIMESTAMPTZ(6),
                title TEXT,
                detail TEXT,
                source_url TEXT,
                raw_payload JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS case_events_saved_case_id_event_date_idx "
            "ON case_events(saved_case_id, event_date);"
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS case_notes (
                id TEXT PRIMARY KEY,
                saved_case_id TEXT NOT NULL REFERENCES saved_cases(id) ON DELETE CASCADE,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                note_text TEXT NOT NULL,
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS case_ai_summaries (
                id TEXT PRIMARY KEY,
                saved_case_id TEXT NOT NULL REFERENCES saved_cases(id) ON DELETE CASCADE,
                summary_text TEXT NOT NULL,
                source_event_id TEXT REFERENCES case_events(id) ON DELETE SET NULL,
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def ensure_notification_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS notification_preferences (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
                email_enabled BOOLEAN NOT NULL DEFAULT true,
                push_enabled BOOLEAN NOT NULL DEFAULT false,
                sms_enabled BOOLEAN NOT NULL DEFAULT false,
                fcm_token TEXT,
                phone_number TEXT,
                type_overrides JSONB NOT NULL DEFAULT '{}'
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS notifications (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                related_case_id TEXT REFERENCES saved_cases(id) ON DELETE SET NULL,
                channel TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                scheduled_for TIMESTAMPTZ(6),
                sent_at TIMESTAMPTZ(6),
                read_at TIMESTAMPTZ(6),
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS notifications_user_id_created_at_idx "
            "ON notifications(user_id, created_at);"
        )


def ensure_cause_list_cache_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS cause_list_cache (
                id TEXT PRIMARY KEY,
                court TEXT NOT NULL,
                list_date TEXT NOT NULL,
                fetched_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                entries JSONB NOT NULL DEFAULT '[]'
            );
            """
        )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS cause_list_cache_court_list_date_key "
            "ON cause_list_cache(court, list_date);"
        )


def ensure_vault_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS vault_documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                document_type TEXT NOT NULL,
                object_key TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                related_case_id TEXT REFERENCES saved_cases(id) ON DELETE SET NULL,
                extracted_text TEXT,
                indexing_status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS vault_document_embeddings (
                id TEXT PRIMARY KEY,
                vault_document_id TEXT NOT NULL REFERENCES vault_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding vector(1024) NOT NULL
            );
            """
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS vault_document_permissions (
                id TEXT PRIMARY KEY,
                vault_document_id TEXT NOT NULL REFERENCES vault_documents(id) ON DELETE CASCADE,
                shared_with_user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                permission TEXT NOT NULL DEFAULT 'view',
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def ensure_chat_tables(engine: Engine) -> None:
    """Per-user chat history: chat_sessions + chat_messages."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "DO $$ BEGIN "
            "CREATE TYPE \"MessageRole\" AS ENUM ('user', 'assistant', 'system'); "
            "EXCEPTION WHEN duplicate_object THEN NULL; END $$;"
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT,
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chat_sessions_user_id_updated_at_idx "
            "ON chat_sessions(user_id, updated_at DESC);"
        )
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role "MessageRole" NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS chat_messages_session_id_created_at_idx "
            "ON chat_messages(session_id, created_at);"
        )


def ensure_calendar_events_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            CREATE TABLE IF NOT EXISTS calendar_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                event_type TEXT NOT NULL,
                start_at TIMESTAMPTZ(6) NOT NULL,
                end_at TIMESTAMPTZ(6),
                related_case_id TEXT REFERENCES saved_cases(id) ON DELETE SET NULL,
                related_booking_id TEXT REFERENCES bookings(id) ON DELETE SET NULL,
                related_case_event_id TEXT REFERENCES case_events(id) ON DELETE SET NULL,
                google_calendar_event_id TEXT,
                outlook_event_id TEXT,
                created_at TIMESTAMPTZ(6) NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS calendar_events_user_id_start_at_idx "
            "ON calendar_events(user_id, start_at);"
        )


def ensure_calendar_events_related_case_event_column(engine: Engine) -> None:
    """Lets sync_case_events() upsert a hearing's calendar_events row instead
    of creating a duplicate on every re-sync (added after calendar_events
    already shipped without this column)."""
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "ALTER TABLE calendar_events ADD COLUMN IF NOT EXISTS "
            "related_case_event_id TEXT REFERENCES case_events(id) ON DELETE SET NULL;"
        )
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "calendar_events_related_case_event_id_key ON calendar_events(related_case_event_id) "
            "WHERE related_case_event_id IS NOT NULL;"
        )


def run_migrations(engine: Engine) -> None:
    """Called unconditionally from init_db() after schema.sql (or on every
    run against an existing DB). Order matters: parent tables/columns first."""
    ensure_lawyer_embedding_column(engine)
    ensure_users_role_column(engine)
    ensure_lawyers_user_id_column(engine)
    ensure_similar_case_searches_table(engine)
    ensure_case_tables(engine)
    ensure_notification_tables(engine)
    ensure_cause_list_cache_table(engine)
    ensure_vault_tables(engine)
    ensure_calendar_events_table(engine)
    ensure_calendar_events_related_case_event_column(engine)
    ensure_chat_tables(engine)
