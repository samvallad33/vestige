//! Source connector trait and complete 8-family store catalog (#775).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The eight universal protocol families in Wave 14N.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConnectorFamily {
    /// Family A: Postgres wire (pgvector, pgvectorscale, Supabase, Neon, AlloyDB, etc.)
    PostgresWire,
    /// Family B: Other SQL (Oracle 23ai, MariaDB, TiDB, ClickHouse, DuckDB VSS, sqlite-vec, Snowflake, etc.)
    SqlVector,
    /// Family C: Lucene & Search APIs (Elasticsearch, OpenSearch, Solr, Vespa, Typesense, Meilisearch, etc.)
    SearchEngine,
    /// Family D: Redis protocol (Redis Stack, RediSearch, Valkey Search, Upstash Vector, Dragonfly, etc.)
    RedisProtocol,
    /// Family E: Native Vector DBs (Pinecone, Weaviate, Qdrant, Milvus, Zilliz, Chroma, LanceDB, Turbopuffer, etc.)
    NativeVectorDb,
    /// Family F: Document & Graph DBs (MongoDB Atlas, Cosmos DB, Cassandra 5, Neo4j, Memgraph, FalkorDB, SurrealDB)
    DocGraphDb,
    /// Family G: Agent Memory Layers (Mem0, Zep, Graphiti, Letta, Cognee, LangGraph, LlamaIndex, CrewAI, CLAUDE.md, AGENTS.md)
    AgentMemoryLayer,
    /// Family H: Streaming Files & Index Formats (JSONL, NDJSON, CSV, Parquet, Arrow IPC, SQLite, FAISS/hnswlib/Annoy + docstore)
    FileAndIndex,
}

impl ConnectorFamily {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::PostgresWire => "A:postgres-wire",
            Self::SqlVector => "B:sql-vector",
            Self::SearchEngine => "C:search-engine",
            Self::RedisProtocol => "D:redis-protocol",
            Self::NativeVectorDb => "E:native-vector-db",
            Self::DocGraphDb => "F:doc-graph-db",
            Self::AgentMemoryLayer => "G:agent-memory-layer",
            Self::FileAndIndex => "H:file-and-index",
        }
    }
}

/// Metadata entry for `vestige import --list-sources`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreCatalogEntry {
    pub slug: &'static str,
    pub display_name: &'static str,
    pub family: ConnectorFamily,
    pub driver: &'static str,
}

/// Complete Wave 14N catalog covering all 8 protocol families.
pub const STORE_CATALOG: &[StoreCatalogEntry] = &[
    // Family A: Postgres wire
    StoreCatalogEntry { slug: "pgvector", display_name: "pgvector", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "pgvectorscale", display_name: "pgvectorscale (Timescale)", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "vectorchord", display_name: "VectorChord / pgvecto.rs", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "lantern", display_name: "Lantern", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "supabase-vector", display_name: "Supabase Vector", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "neon", display_name: "Neon PostgreSQL", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "aurora-pg", display_name: "Amazon Aurora / RDS PostgreSQL", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "alloydb", display_name: "Google AlloyDB", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "cloudsql-pg", display_name: "Cloud SQL for PostgreSQL", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "azure-pg", display_name: "Azure Database for PostgreSQL", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "yugabytedb", display_name: "YugabyteDB", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "cockroachdb", display_name: "CockroachDB Vector", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    StoreCatalogEntry { slug: "paradedb", display_name: "ParadeDB", family: ConnectorFamily::PostgresWire, driver: "postgres-readonly-sql" },
    // Family B: Other SQL
    StoreCatalogEntry { slug: "oracle-23ai", display_name: "Oracle Database 23ai AI Vector Search", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "mariadb-vector", display_name: "MariaDB Vector", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "mysql-heatwave", display_name: "MySQL HeatWave", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "tidb-vector", display_name: "TiDB Vector", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "singlestore", display_name: "SingleStore", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "clickhouse", display_name: "ClickHouse", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "duckdb-vss", display_name: "DuckDB VSS", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "sqlite-vec", display_name: "SQLite sqlite-vec / sqlite-vss", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "turso", display_name: "libSQL / Turso", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "doris", display_name: "Apache Doris", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "starrocks", display_name: "StarRocks", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "snowflake-cortex", display_name: "Snowflake VECTOR / Cortex Search", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "databricks-mosaic", display_name: "Databricks Mosaic AI Vector Search", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "bigquery-vector", display_name: "BigQuery Vector Search", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "spanner", display_name: "Google Spanner", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    StoreCatalogEntry { slug: "teradata", display_name: "Teradata", family: ConnectorFamily::SqlVector, driver: "sql-readonly-cursor" },
    // Family C: Lucene & Search Engine APIs
    StoreCatalogEntry { slug: "elasticsearch", display_name: "Elasticsearch", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "opensearch", display_name: "OpenSearch / Amazon OpenSearch", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "solr", display_name: "Apache Solr", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "vespa", display_name: "Vespa", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "typesense", display_name: "Typesense", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "meilisearch", display_name: "Meilisearch", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "manticore", display_name: "Manticore Search", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "azure-ai-search", display_name: "Azure AI Search", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    StoreCatalogEntry { slug: "marqo", display_name: "Marqo", family: ConnectorFamily::SearchEngine, driver: "pit-scroll-reader" },
    // Family D: Redis protocol
    StoreCatalogEntry { slug: "redis-stack", display_name: "Redis Stack / RediSearch / Redis Cloud", family: ConnectorFamily::RedisProtocol, driver: "redis-scan-ft" },
    StoreCatalogEntry { slug: "valkey-search", display_name: "Valkey Search", family: ConnectorFamily::RedisProtocol, driver: "redis-scan-ft" },
    StoreCatalogEntry { slug: "memorydb", display_name: "Amazon MemoryDB", family: ConnectorFamily::RedisProtocol, driver: "redis-scan-ft" },
    StoreCatalogEntry { slug: "upstash-vector", display_name: "Upstash Vector", family: ConnectorFamily::RedisProtocol, driver: "redis-scan-ft" },
    StoreCatalogEntry { slug: "dragonfly", display_name: "Dragonfly", family: ConnectorFamily::RedisProtocol, driver: "redis-scan-ft" },
    // Family E: Native Vector DBs
    StoreCatalogEntry { slug: "pinecone", display_name: "Pinecone (Serverless & Pod)", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "weaviate", display_name: "Weaviate", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "qdrant", display_name: "Qdrant", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "milvus", display_name: "Milvus / Zilliz Cloud", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "chroma", display_name: "Chroma / Chroma Cloud", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "lancedb", display_name: "LanceDB / LanceDB Cloud", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "turbopuffer", display_name: "Turbopuffer", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "vald", display_name: "Vald", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "vearch", display_name: "Vearch", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "deeplake", display_name: "Deep Lake (Activeloop)", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "vectara", display_name: "Vectara", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "epsilla", display_name: "Epsilla", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "myscale", display_name: "MyScale", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "astra-db", display_name: "Astra DB (DataStax)", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "kdb-ai", display_name: "KDB.AI", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "infinity", display_name: "Infinity (InfiniFlow)", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "aerospike-vector", display_name: "Aerospike Vector Search", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "vertex-vector", display_name: "Google Vertex AI Vector Search", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "s3-vectors", display_name: "Amazon S3 Vectors", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "bedrock-kb", display_name: "Amazon Bedrock Knowledge Bases", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "kendra", display_name: "Amazon Kendra", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    StoreCatalogEntry { slug: "openai-vector-store", display_name: "OpenAI Vector Stores (File Search)", family: ConnectorFamily::NativeVectorDb, driver: "paged-vector-rest" },
    // Family F: Document & Graph DBs
    StoreCatalogEntry { slug: "mongodb-atlas", display_name: "MongoDB Atlas Vector Search", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "cosmos-db", display_name: "Azure Cosmos DB (NoSQL & vCore)", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "documentdb", display_name: "Amazon DocumentDB", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "couchbase", display_name: "Couchbase", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "firestore-vector", display_name: "Firestore Vector Search", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "cassandra-sai", display_name: "Apache Cassandra 5 SAI", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "scylladb", display_name: "ScyllaDB", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "neo4j", display_name: "Neo4j Vector Index", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "memgraph", display_name: "Memgraph", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "falkordb", display_name: "FalkorDB", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "arangodb", display_name: "ArangoDB", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    StoreCatalogEntry { slug: "surrealdb", display_name: "SurrealDB", family: ConnectorFamily::DocGraphDb, driver: "doc-graph-cursor" },
    // Family G: Memory layers & agent memory
    StoreCatalogEntry { slug: "mem0", display_name: "Mem0", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "zep", display_name: "Zep / Graphiti", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "letta", display_name: "Letta (MemGPT)", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "cognee", display_name: "Cognee", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "supermemory", display_name: "Supermemory", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "kernel-memory", display_name: "Microsoft Kernel Memory", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "langgraph-store", display_name: "LangGraph Store / LangChain Memory", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "llamaindex-docstore", display_name: "LlamaIndex Vector & Doc Stores", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "haystack", display_name: "Haystack Document Stores", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "crewai-memory", display_name: "CrewAI Memory", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "autogen-memory", display_name: "AutoGen Memory", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "semantic-kernel", display_name: "Semantic Kernel Memory", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "pinecone-assistant", display_name: "Pinecone Assistant", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "txtai", display_name: "txtai", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-export" },
    StoreCatalogEntry { slug: "claude-memory", display_name: "Claude Code Memory (CLAUDE.md / .claude/)", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-files" },
    StoreCatalogEntry { slug: "cursor-rules", display_name: "Cursor Rules (.cursorrules / .cursor/rules)", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-files" },
    StoreCatalogEntry { slug: "agents-md", display_name: "AGENTS.md / GEMINI.md Rules", family: ConnectorFamily::AgentMemoryLayer, driver: "agent-memory-files" },
    // Family H: Streaming files & index formats
    StoreCatalogEntry { slug: "faiss", display_name: "FAISS Index + Sidecar Docstore", family: ConnectorFamily::FileAndIndex, driver: "index-sidecar-reader" },
    StoreCatalogEntry { slug: "hnswlib", display_name: "hnswlib Index + Sidecar Docstore", family: ConnectorFamily::FileAndIndex, driver: "index-sidecar-reader" },
    StoreCatalogEntry { slug: "annoy", display_name: "Annoy Index + Sidecar Docstore", family: ConnectorFamily::FileAndIndex, driver: "index-sidecar-reader" },
    StoreCatalogEntry { slug: "scann", display_name: "ScaNN Index + Sidecar Docstore", family: ConnectorFamily::FileAndIndex, driver: "index-sidecar-reader" },
    StoreCatalogEntry { slug: "usearch", display_name: "USearch Index + Sidecar Docstore", family: ConnectorFamily::FileAndIndex, driver: "index-sidecar-reader" },
    StoreCatalogEntry { slug: "voyager", display_name: "Voyager Index + Sidecar Docstore", family: ConnectorFamily::FileAndIndex, driver: "index-sidecar-reader" },
    StoreCatalogEntry { slug: "lance-file", display_name: "Lance Dataset File", family: ConnectorFamily::FileAndIndex, driver: "streaming-file-reader" },
    StoreCatalogEntry { slug: "parquet", display_name: "Apache Parquet", family: ConnectorFamily::FileAndIndex, driver: "streaming-file-reader" },
    StoreCatalogEntry { slug: "arrow-ipc", display_name: "Apache Arrow IPC", family: ConnectorFamily::FileAndIndex, driver: "streaming-file-reader" },
    StoreCatalogEntry { slug: "jsonl", display_name: "JSONL / NDJSON Export", family: ConnectorFamily::FileAndIndex, driver: "streaming-file-reader" },
    StoreCatalogEntry { slug: "csv", display_name: "CSV Export", family: ConnectorFamily::FileAndIndex, driver: "streaming-file-reader" },
    StoreCatalogEntry { slug: "sqlite-file", display_name: "SQLite Database Export", family: ConnectorFamily::FileAndIndex, driver: "streaming-file-reader" },
];

pub fn lookup_catalog(slug: &str) -> Option<&'static StoreCatalogEntry> {
    STORE_CATALOG.iter().find(|e| e.slug.eq_ignore_ascii_case(slug))
}

/// Raw record pulled from an external source before vector-stripping & anchoring.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawSourceRecord {
    pub source_id: String,
    pub text: Option<String>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub commit_ref: Option<String>,
    #[serde(default)]
    pub file_paths: Vec<String>,
    #[serde(default)]
    pub cites: Vec<String>,
}

/// Explicit read error returned by a connector (#496: never silently defaulted).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorReadError {
    pub code: String,
    pub detail: String,
}

/// Page of records plus an optional resume cursor.
#[derive(Debug, Clone)]
pub struct PullPage {
    pub records: Vec<RawSourceRecord>,
    pub next_cursor: Option<String>,
}

/// Universal read-only connector trait implemented across all 8 protocol families.
pub trait SourceConnector {
    fn slug(&self) -> &str;
    fn family(&self) -> ConnectorFamily;
    fn version(&self) -> &str {
        "6.2.3"
    }
    fn list_collections(&self) -> Result<Vec<String>, ConnectorReadError>;
    fn count(&self, collection: &str) -> Result<Option<u64>, ConnectorReadError>;
    fn pull_page(
        &self,
        collection: &str,
        cursor: Option<&str>,
    ) -> Result<PullPage, ConnectorReadError>;
    fn snapshot_digest(&self, collection: &str) -> Result<String, ConnectorReadError>;
}
