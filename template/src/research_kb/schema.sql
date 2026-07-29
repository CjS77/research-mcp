-- research-kb schema. The vec0 virtual table is created in db.py so its
-- dimension can follow KB_EMBED_DIM. Everything dimension-independent lives here.

PRAGMA foreign_keys = ON;

-- 3.1 Documents -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    source_path TEXT UNIQUE NOT NULL,
    doc_type TEXT NOT NULL,                 -- 'paper','research','assessment','sketch','spec'
    tier TEXT NOT NULL DEFAULT 'breadth',   -- 'core' (validated, load-bearing) | 'breadth'
    title TEXT NOT NULL,
    phase INTEGER,                          -- 1/2/3 for staged research docs, NULL for source works
    facet_b TEXT,                           -- JSON array (secondary facet)
    facet_a TEXT,                           -- JSON array (primary facet)
    authors TEXT,                           -- JSON array
    year INTEGER,
    page_count INTEGER,
    validated INTEGER DEFAULT 0,            -- human sign-off (core tier)
    validated_at TIMESTAMP,
    content_hash TEXT,                      -- SHA-256, change detection
    word_count INTEGER,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_tier ON documents(tier);
CREATE INDEX IF NOT EXISTS idx_documents_phase ON documents(phase);

-- 3.2 Chunks (with provenance) ---------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_kind TEXT NOT NULL,             -- 'verbatim' | 'derived_summary' | 'enrichment'
    section_number TEXT,                    -- '1.2', '3.3.1'
    section_title TEXT,
    chunk_type TEXT NOT NULL,               -- paragraph/table/code/math/theorem/protocol/heading
    page_start INTEGER,
    page_end INTEGER,
    verbatim_hash TEXT,                     -- hash of source span, for audit
    parent_chunk_id INTEGER,                -- hierarchical retrieval
    token_count INTEGER,
    embed_input TEXT,                       -- contextualized text actually embedded
    embedded INTEGER NOT NULL DEFAULT 0,    -- 1 when a vector exists in vec_chunks
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(content_kind);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);

-- 3.3 Full-text search (FTS5) ----------------------------------------------
-- External-content table mirroring chunks(content, section_title).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    section_title,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Keep chunks_fts in sync with chunks.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, section_title)
    VALUES (new.id, new.content, new.section_title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, section_title)
    VALUES ('delete', old.id, old.content, old.section_title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, section_title)
    VALUES ('delete', old.id, old.content, old.section_title);
    INSERT INTO chunks_fts(rowid, content, section_title)
    VALUES (new.id, new.content, new.section_title);
END;

-- 3.4 Citations (graph edges) ----------------------------------------------
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY,
    from_document_id INTEGER NOT NULL,
    to_reference TEXT NOT NULL,             -- raw cited-work string as it appears
    to_document_id INTEGER,                 -- resolved when cited work is in-corpus, else NULL
    context_snippet TEXT,
    page_ref INTEGER,
    FOREIGN KEY (from_document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (to_document_id)   REFERENCES documents(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_citations_from ON citations(from_document_id);
CREATE INDEX IF NOT EXISTS idx_citations_to ON citations(to_document_id);

-- 3.5 Eval queries & indexing jobs -----------------------------------------
CREATE TABLE IF NOT EXISTS eval_queries (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    expected_document_id INTEGER,
    expected_section TEXT,
    expected_page INTEGER,
    notes TEXT,
    FOREIGN KEY (expected_document_id) REFERENCES documents(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS indexing_jobs (
    id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL,                   -- pending/processing/completed/failed
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    chunks_created INTEGER
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON indexing_jobs(status);
