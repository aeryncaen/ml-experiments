// Package state manages per-shard pipeline state in SQLite.
//
// Every shard transition (pending → running → done/failed) is recorded so the
// pipeline can be killed at any point and resume where it left off.
package state

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// Stage is a pipeline processing stage.
type Stage string

const (
	StageDownload Stage = "download"
	StageIngest   Stage = "ingest"
	StageProcess  Stage = "process"
	StageTokenize Stage = "tokenize"
)

// Status is the processing status of a shard.
type Status string

const (
	StatusPending Status = "pending"
	StatusRunning Status = "running"
	StatusDone    Status = "done"
	StatusFailed  Status = "failed"
)

// Shard represents a single shard's state in the database.
type Shard struct {
	Dataset    string
	ShardIdx   int
	Stage      Stage
	Status     Status
	InputPath  string
	OutputPath string
	Tokens     int64
	Docs       int64
	Bytes      int64
	Error      string
	StartedAt  string
	FinishedAt string
}

// RunRecord is a pipeline execution log entry.
type RunRecord struct {
	ID         int64
	StartedAt  string
	FinishedAt string
	ConfigHash string
	Command    string
}

// DB wraps a SQLite connection for pipeline state tracking.
type DB struct {
	db *sql.DB
}

// Open opens (or creates) the state database at the given path.
// It creates tables if they don't exist and resets any interrupted shards.
func Open(path string) (*DB, error) {
	sqldb, err := sql.Open("sqlite3", path+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("open state db: %w", err)
	}

	// Single connection for writes to avoid SQLITE_BUSY in WAL mode.
	sqldb.SetMaxOpenConns(1)

	s := &DB{db: sqldb}
	if err := s.migrate(); err != nil {
		sqldb.Close()
		return nil, fmt.Errorf("migrate state db: %w", err)
	}
	if err := s.resetInterrupted(); err != nil {
		sqldb.Close()
		return nil, fmt.Errorf("reset interrupted shards: %w", err)
	}
	return s, nil
}

// Close closes the database connection.
func (s *DB) Close() error {
	return s.db.Close()
}

func (s *DB) migrate() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS shards (
			dataset     TEXT    NOT NULL,
			shard_idx   INTEGER NOT NULL,
			stage       TEXT    NOT NULL,
			status      TEXT    DEFAULT 'pending',
			input_path  TEXT,
			output_path TEXT,
			tokens      INTEGER DEFAULT 0,
			docs        INTEGER DEFAULT 0,
			bytes       INTEGER DEFAULT 0,
			error       TEXT,
			started_at  TEXT,
			finished_at TEXT,
			PRIMARY KEY (dataset, shard_idx, stage)
		);

		CREATE TABLE IF NOT EXISTS runs (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			started_at  TEXT    NOT NULL,
			finished_at TEXT,
			config_hash TEXT,
			command     TEXT
		);

		CREATE INDEX IF NOT EXISTS idx_shards_status
			ON shards(dataset, stage, status);
	`)
	return err
}

// resetInterrupted marks any running shards back to pending.
// Called on startup so interrupted work gets retried.
func (s *DB) resetInterrupted() error {
	res, err := s.db.Exec(`UPDATE shards SET status = 'pending', started_at = NULL WHERE status = 'running'`)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n > 0 {
		fmt.Printf("[state] reset %d interrupted shards to pending\n", n)
	}
	return nil
}

// EnsureShard creates a shard record if it doesn't already exist.
// Returns true if a new record was inserted, false if it already existed.
func (s *DB) EnsureShard(dataset string, shardIdx int, stage Stage, inputPath string) (bool, error) {
	res, err := s.db.Exec(`
		INSERT OR IGNORE INTO shards (dataset, shard_idx, stage, status, input_path)
		VALUES (?, ?, ?, 'pending', ?)
	`, dataset, shardIdx, string(stage), inputPath)
	if err != nil {
		return false, err
	}
	n, _ := res.RowsAffected()
	return n > 0, nil
}

// EnsureShardsBatch creates multiple shard records in a single transaction.
// Ignores duplicates.
func (s *DB) EnsureShardsBatch(shards []Shard) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT OR IGNORE INTO shards (dataset, shard_idx, stage, status, input_path)
		VALUES (?, ?, ?, 'pending', ?)
	`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for _, sh := range shards {
		if _, err := stmt.Exec(sh.Dataset, sh.ShardIdx, string(sh.Stage), sh.InputPath); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// MarkRunning marks a shard as currently being processed.
func (s *DB) MarkRunning(dataset string, shardIdx int, stage Stage) error {
	_, err := s.db.Exec(`
		UPDATE shards SET status = 'running', started_at = ?
		WHERE dataset = ? AND shard_idx = ? AND stage = ?
	`, nowUTC(), dataset, shardIdx, string(stage))
	return err
}

// MarkDone marks a shard as successfully completed with stats.
func (s *DB) MarkDone(dataset string, shardIdx int, stage Stage, outputPath string, tokens, docs, bytes int64) error {
	_, err := s.db.Exec(`
		UPDATE shards SET status = 'done', output_path = ?, tokens = ?, docs = ?, bytes = ?, finished_at = ?, error = NULL
		WHERE dataset = ? AND shard_idx = ? AND stage = ?
	`, outputPath, tokens, docs, bytes, nowUTC(), dataset, shardIdx, string(stage))
	return err
}

// MarkFailed marks a shard as failed with an error message.
func (s *DB) MarkFailed(dataset string, shardIdx int, stage Stage, errMsg string) error {
	_, err := s.db.Exec(`
		UPDATE shards SET status = 'failed', error = ?, finished_at = ?
		WHERE dataset = ? AND shard_idx = ? AND stage = ?
	`, errMsg, nowUTC(), dataset, shardIdx, string(stage))
	return err
}

// PendingShards returns all pending shards for a dataset+stage, ordered by shard index.
func (s *DB) PendingShards(dataset string, stage Stage) ([]Shard, error) {
	return s.queryShards(`
		SELECT dataset, shard_idx, stage, status, input_path, output_path, tokens, docs, bytes, error, started_at, finished_at
		FROM shards WHERE dataset = ? AND stage = ? AND status = 'pending'
		ORDER BY shard_idx
	`, dataset, string(stage))
}

// AllShards returns all shards for a dataset+stage, ordered by shard index.
func (s *DB) AllShards(dataset string, stage Stage) ([]Shard, error) {
	return s.queryShards(`
		SELECT dataset, shard_idx, stage, status, input_path, output_path, tokens, docs, bytes, error, started_at, finished_at
		FROM shards WHERE dataset = ? AND stage = ?
		ORDER BY shard_idx
	`, dataset, string(stage))
}

// ShardsByStatus returns shards for a dataset+stage filtered by status.
func (s *DB) ShardsByStatus(dataset string, stage Stage, status Status) ([]Shard, error) {
	return s.queryShards(`
		SELECT dataset, shard_idx, stage, status, input_path, output_path, tokens, docs, bytes, error, started_at, finished_at
		FROM shards WHERE dataset = ? AND stage = ? AND status = ?
		ORDER BY shard_idx
	`, dataset, string(stage), string(status))
}

// DatasetStageStatus is a summary of shard counts by status for one dataset+stage.
type DatasetStageStatus struct {
	Dataset string
	Stage   Stage
	Total   int
	Pending int
	Running int
	Done    int
	Failed  int
	Tokens  int64
	Docs    int64
}

// StatusSummary returns per-dataset, per-stage shard counts.
// If datasets is non-empty, only those datasets are included.
func (s *DB) StatusSummary(datasets []string) ([]DatasetStageStatus, error) {
	query := `
		SELECT dataset, stage,
			COUNT(*) as total,
			SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
			SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
			SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as done,
			SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
			SUM(tokens) as tokens,
			SUM(docs) as docs
		FROM shards
	`
	var args []any
	if len(datasets) > 0 {
		placeholders := make([]string, len(datasets))
		for i, d := range datasets {
			placeholders[i] = "?"
			args = append(args, d)
		}
		query += " WHERE dataset IN (" + strings.Join(placeholders, ",") + ")"
	}
	query += " GROUP BY dataset, stage ORDER BY dataset, stage"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []DatasetStageStatus
	for rows.Next() {
		var ds DatasetStageStatus
		var stageStr string
		if err := rows.Scan(&ds.Dataset, &stageStr, &ds.Total, &ds.Pending, &ds.Running, &ds.Done, &ds.Failed, &ds.Tokens, &ds.Docs); err != nil {
			return nil, err
		}
		ds.Stage = Stage(stageStr)
		results = append(results, ds)
	}
	return results, rows.Err()
}

// ResetDataset deletes all shard records for a dataset, optionally scoped to a stage.
// If stage is empty, all stages are reset.
func (s *DB) ResetDataset(dataset string, stage Stage) (int64, error) {
	var res sql.Result
	var err error
	if stage == "" {
		res, err = s.db.Exec(`DELETE FROM shards WHERE dataset = ?`, dataset)
	} else {
		res, err = s.db.Exec(`DELETE FROM shards WHERE dataset = ? AND stage = ?`, dataset, string(stage))
	}
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// ResetFailed resets all failed shards for a dataset+stage back to pending.
func (s *DB) ResetFailed(dataset string, stage Stage) (int64, error) {
	var res sql.Result
	var err error
	if stage == "" {
		res, err = s.db.Exec(`UPDATE shards SET status = 'pending', error = NULL, started_at = NULL, finished_at = NULL WHERE dataset = ? AND status = 'failed'`, dataset)
	} else {
		res, err = s.db.Exec(`UPDATE shards SET status = 'pending', error = NULL, started_at = NULL, finished_at = NULL WHERE dataset = ? AND stage = ? AND status = 'failed'`, dataset, string(stage))
	}
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// StartRun records the start of a pipeline run.
func (s *DB) StartRun(configHash, command string) (int64, error) {
	res, err := s.db.Exec(`INSERT INTO runs (started_at, config_hash, command) VALUES (?, ?, ?)`,
		nowUTC(), configHash, command)
	if err != nil {
		return 0, err
	}
	return res.LastInsertId()
}

// FinishRun marks a run as completed.
func (s *DB) FinishRun(runID int64) error {
	_, err := s.db.Exec(`UPDATE runs SET finished_at = ? WHERE id = ?`, nowUTC(), runID)
	return err
}

// TotalTokens returns the sum of tokens across all done shards for a dataset+stage.
func (s *DB) TotalTokens(dataset string, stage Stage) (int64, error) {
	var total sql.NullInt64
	err := s.db.QueryRow(`
		SELECT SUM(tokens) FROM shards
		WHERE dataset = ? AND stage = ? AND status = 'done'
	`, dataset, string(stage)).Scan(&total)
	if err != nil {
		return 0, err
	}
	return total.Int64, nil
}

func (s *DB) queryShards(query string, args ...any) ([]Shard, error) {
	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var shards []Shard
	for rows.Next() {
		var sh Shard
		var stageStr, statusStr string
		var inputPath, outputPath, errStr, startedAt, finishedAt sql.NullString
		if err := rows.Scan(
			&sh.Dataset, &sh.ShardIdx, &stageStr, &statusStr,
			&inputPath, &outputPath,
			&sh.Tokens, &sh.Docs, &sh.Bytes,
			&errStr, &startedAt, &finishedAt,
		); err != nil {
			return nil, err
		}
		sh.Stage = Stage(stageStr)
		sh.Status = Status(statusStr)
		sh.InputPath = inputPath.String
		sh.OutputPath = outputPath.String
		sh.Error = errStr.String
		sh.StartedAt = startedAt.String
		sh.FinishedAt = finishedAt.String
		shards = append(shards, sh)
	}
	return shards, rows.Err()
}

func nowUTC() string {
	return time.Now().UTC().Format(time.RFC3339)
}
