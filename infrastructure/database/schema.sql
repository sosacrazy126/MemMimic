-- MemMimic PostgreSQL Schema with Sharding Support
-- Production-ready database schema for horizontal scaling

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Shard configuration table (exists on coordinator)
CREATE TABLE IF NOT EXISTS shard_config (
    shard_id INTEGER PRIMARY KEY,
    shard_name VARCHAR(50) NOT NULL UNIQUE,
    connection_string TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Memory tables (replicated across shards)
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    memory_type VARCHAR(50) DEFAULT 'interaction',
    importance_score REAL DEFAULT 0.5,
    metadata JSONB DEFAULT '{}',
    cxd_classification VARCHAR(20),
    cxd_confidence REAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    shard_key VARCHAR(100) -- for sharding logic
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance_score DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_cxd ON memories(cxd_classification);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(is_active) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_memories_shard_key ON memories(shard_key);
CREATE INDEX IF NOT EXISTS idx_memories_content_gin ON memories USING gin(to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_memories_metadata_gin ON memories USING gin(metadata);

-- Memory vectors table for semantic search
CREATE TABLE IF NOT EXISTS memory_vectors (
    memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL,
    vector_data BYTEA, -- Stored as binary for efficiency
    vector_dimension INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_vectors_model ON memory_vectors(embedding_model);

-- Tales management (centralized, not sharded)
CREATE TABLE IF NOT EXISTS tales (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100) DEFAULT 'misc/general',
    content TEXT NOT NULL,
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tales_name_category ON tales(name, category);
CREATE INDEX IF NOT EXISTS idx_tales_category ON tales(category);
CREATE INDEX IF NOT EXISTS idx_tales_tags_gin ON tales USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_tales_content_gin ON tales USING gin(to_tsvector('english', content));

-- Quality gate queue
CREATE TABLE IF NOT EXISTS quality_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content TEXT NOT NULL,
    memory_type VARCHAR(50) DEFAULT 'interaction',
    metadata JSONB DEFAULT '{}',
    quality_score REAL,
    similarity_score REAL,
    auto_approve_eligible BOOLEAN DEFAULT false,
    status VARCHAR(20) DEFAULT 'pending', -- pending, approved, rejected
    reviewer_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    reviewed_at TIMESTAMP WITH TIME ZONE,
    shard_key VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_quality_queue_status ON quality_queue(status);
CREATE INDEX IF NOT EXISTS idx_quality_queue_created_at ON quality_queue(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_quality_queue_shard_key ON quality_queue(shard_key);

-- Memory analytics and metrics
CREATE TABLE IF NOT EXISTS memory_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    labels JSONB DEFAULT '{}',
    shard_id INTEGER,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_metrics_type_name ON memory_metrics(metric_type, metric_name);
CREATE INDEX IF NOT EXISTS idx_memory_metrics_recorded_at ON memory_metrics(recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_metrics_shard ON memory_metrics(shard_id);

-- Consciousness state tracking
CREATE TABLE IF NOT EXISTS consciousness_state (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    state_type VARCHAR(50) NOT NULL,
    state_data JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_consciousness_state_type ON consciousness_state(state_type);
CREATE INDEX IF NOT EXISTS idx_consciousness_state_updated ON consciousness_state(updated_at DESC);

-- Partitioning strategy for memories table (by month)
CREATE TABLE memories_y2025m01 PARTITION OF memories
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE memories_y2025m02 PARTITION OF memories
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE memories_y2025m03 PARTITION OF memories
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE memories_y2025m04 PARTITION OF memories
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE memories_y2025m05 PARTITION OF memories
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE memories_y2025m06 PARTITION OF memories
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE memories_y2025m07 PARTITION OF memories
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE memories_y2025m08 PARTITION OF memories
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE memories_y2025m09 PARTITION OF memories
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE memories_y2025m10 PARTITION OF memories
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE memories_y2025m11 PARTITION OF memories
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE memories_y2025m12 PARTITION OF memories
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Functions for automatic partition creation
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS VOID AS $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
BEGIN
    -- Create partitions for next 12 months
    FOR i IN 0..11 LOOP
        start_date := date_trunc('month', NOW() + INTERVAL '1 month' * i);
        end_date := start_date + INTERVAL '1 month';
        partition_name := 'memories_y' || to_char(start_date, 'YYYY') || 'm' || to_char(start_date, 'MM');
        
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF memories FOR VALUES FROM (%L) TO (%L)',
                      partition_name, start_date, end_date);
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE TRIGGER update_memories_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tales_updated_at BEFORE UPDATE ON tales
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_consciousness_state_updated_at BEFORE UPDATE ON consciousness_state
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sharding functions
CREATE OR REPLACE FUNCTION get_shard_for_key(shard_key TEXT)
RETURNS INTEGER AS $$
BEGIN
    -- Simple hash-based sharding (can be replaced with more sophisticated logic)
    RETURN (hashtext(shard_key) % 4) + 1; -- 4 shards for now
END;
$$ LANGUAGE plpgsql;

-- Memory cleanup function
CREATE OR REPLACE FUNCTION cleanup_old_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- Archive memories older than 1 year with low importance
    UPDATE memories 
    SET is_active = false 
    WHERE created_at < NOW() - INTERVAL '1 year' 
      AND importance_score < 0.3 
      AND is_active = true;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Actually delete memories older than 2 years
    DELETE FROM memories 
    WHERE created_at < NOW() - INTERVAL '2 years' 
      AND importance_score < 0.2;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring views
CREATE OR REPLACE VIEW memory_stats AS
SELECT 
    memory_type,
    cxd_classification,
    COUNT(*) as total_memories,
    AVG(importance_score) as avg_importance,
    AVG(access_count) as avg_access_count,
    MIN(created_at) as oldest_memory,
    MAX(created_at) as newest_memory
FROM memories 
WHERE is_active = true
GROUP BY memory_type, cxd_classification;

CREATE OR REPLACE VIEW shard_distribution AS
SELECT 
    shard_key,
    COUNT(*) as memory_count,
    AVG(importance_score) as avg_importance
FROM memories 
WHERE is_active = true
GROUP BY shard_key
ORDER BY memory_count DESC;

-- Initialize shard configuration
INSERT INTO shard_config (shard_id, shard_name, connection_string) 
VALUES 
    (1, 'shard-001', 'postgresql://memmimic:password@postgres-shard-1:5432/memmimic_shard_1'),
    (2, 'shard-002', 'postgresql://memmimic:password@postgres-shard-2:5432/memmimic_shard_2'),
    (3, 'shard-003', 'postgresql://memmimic:password@postgres-shard-3:5432/memmimic_shard_3'),
    (4, 'shard-004', 'postgresql://memmimic:password@postgres-shard-4:5432/memmimic_shard_4')
ON CONFLICT (shard_id) DO NOTHING;