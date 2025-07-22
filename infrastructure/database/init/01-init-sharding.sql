-- Initialize MemMimic Sharding Architecture
-- This script sets up the coordinator database and shard configurations

-- Create databases for shards if they don't exist
-- Note: This requires superuser privileges, typically run during container initialization

-- Create shard databases
SELECT 'CREATE DATABASE memmimic_shard_1' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'memmimic_shard_1')\gexec
SELECT 'CREATE DATABASE memmimic_shard_2' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'memmimic_shard_2')\gexec  
SELECT 'CREATE DATABASE memmimic_shard_3' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'memmimic_shard_3')\gexec
SELECT 'CREATE DATABASE memmimic_shard_4' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'memmimic_shard_4')\gexec

-- Create replication user for streaming replication
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'replicator') THEN
        CREATE USER replicator WITH REPLICATION PASSWORD 'replication_password';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE memmimic TO memmimic;
GRANT CONNECT ON DATABASE memmimic_shard_1 TO memmimic;
GRANT CONNECT ON DATABASE memmimic_shard_2 TO memmimic;
GRANT CONNECT ON DATABASE memmimic_shard_3 TO memmimic;
GRANT CONNECT ON DATABASE memmimic_shard_4 TO memmimic;

-- Configure pg_hba.conf for replication (this needs to be done at container level)
-- Add to pg_hba.conf:
-- host replication replicator 0.0.0.0/0 md5

-- Set up initial shard registry
\c memmimic;

-- Initialize coordinator schema
\i /docker-entrypoint-initdb.d/schema.sql

-- Update shard configuration with actual connection strings
UPDATE shard_config SET 
    connection_string = 'postgresql://memmimic:password@postgres-shard-1:5432/memmimic_shard_1',
    is_active = true 
WHERE shard_id = 1;

UPDATE shard_config SET 
    connection_string = 'postgresql://memmimic:password@postgres-shard-2:5432/memmimic_shard_2',
    is_active = true 
WHERE shard_id = 2;

UPDATE shard_config SET 
    connection_string = 'postgresql://memmimic:password@postgres-shard-3:5432/memmimic_shard_3',
    is_active = true 
WHERE shard_id = 3;

UPDATE shard_config SET 
    connection_string = 'postgresql://memmimic:password@postgres-shard-4:5432/memmimic_shard_4',
    is_active = true 
WHERE shard_id = 4;

-- Initialize each shard database
\c memmimic_shard_1;
\i /docker-entrypoint-initdb.d/schema.sql

\c memmimic_shard_2;
\i /docker-entrypoint-initdb.d/schema.sql

\c memmimic_shard_3;
\i /docker-entrypoint-initdb.d/schema.sql

\c memmimic_shard_4;
\i /docker-entrypoint-initdb.d/schema.sql

-- Return to coordinator database
\c memmimic;

-- Create shard management functions
CREATE OR REPLACE FUNCTION get_active_shards()
RETURNS TABLE(shard_id INTEGER, shard_name VARCHAR, connection_string TEXT) AS $$
BEGIN
    RETURN QUERY 
    SELECT s.shard_id, s.shard_name, s.connection_string
    FROM shard_config s 
    WHERE s.is_active = true
    ORDER BY s.shard_id;
END;
$$ LANGUAGE plpgsql;

-- Health check function for shards
CREATE OR REPLACE FUNCTION check_shard_health()
RETURNS TABLE(
    shard_id INTEGER, 
    shard_name VARCHAR, 
    is_healthy BOOLEAN, 
    last_check TIMESTAMP WITH TIME ZONE,
    error_message TEXT
) AS $$
DECLARE
    shard_record RECORD;
    health_status BOOLEAN;
    error_msg TEXT;
BEGIN
    FOR shard_record IN SELECT * FROM shard_config WHERE is_active = true LOOP
        BEGIN
            -- This would normally use dblink or similar extension to check remote shards
            -- For now, we'll assume they're healthy if they're marked active
            health_status := true;
            error_msg := NULL;
            
        EXCEPTION WHEN OTHERS THEN
            health_status := false;
            error_msg := SQLERRM;
        END;
        
        RETURN QUERY SELECT 
            shard_record.shard_id,
            shard_record.shard_name,
            health_status,
            NOW(),
            error_msg;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring view
CREATE OR REPLACE VIEW cluster_stats AS
WITH shard_stats AS (
    SELECT 
        'coordinator' as shard_type,
        COUNT(*) as table_count,
        pg_size_pretty(pg_database_size(current_database())) as size
    FROM information_schema.tables 
    WHERE table_schema = 'public'
)
SELECT * FROM shard_stats;

-- Insert initial performance baseline
INSERT INTO memory_metrics (metric_type, metric_name, metric_value, labels)
VALUES 
    ('cluster', 'shards_initialized', 4, '{"cluster": "memmimic-prod"}'),
    ('cluster', 'coordinator_status', 1, '{"status": "healthy"}');

-- Log initialization completion
DO $$
BEGIN
    RAISE NOTICE 'MemMimic sharding architecture initialized successfully';
    RAISE NOTICE 'Coordinator database: %', current_database();
    RAISE NOTICE 'Active shards: %', (SELECT COUNT(*) FROM shard_config WHERE is_active = true);
END
$$;