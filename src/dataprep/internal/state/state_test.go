package state

import (
	"os"
	"path/filepath"
	"testing"
)

func tempDB(t *testing.T) *DB {
	t.Helper()
	dir := t.TempDir()
	db, err := Open(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("open test db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func TestOpenAndMigrate(t *testing.T) {
	db := tempDB(t)
	// Should be able to open again (idempotent migration).
	dir := t.TempDir()
	path := filepath.Join(dir, "test2.db")
	db2, err := Open(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	db2.Close()

	// Re-open should work.
	db3, err := Open(path)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	db3.Close()
	_ = db
}

func TestEnsureShardAndQuery(t *testing.T) {
	db := tempDB(t)

	// Insert a shard.
	inserted, err := db.EnsureShard("test-ds", 0, StageIngest, "/input/0.parquet")
	if err != nil {
		t.Fatal(err)
	}
	if !inserted {
		t.Error("expected insert")
	}

	// Insert same shard again — should be no-op.
	inserted, err = db.EnsureShard("test-ds", 0, StageIngest, "/input/0.parquet")
	if err != nil {
		t.Fatal(err)
	}
	if inserted {
		t.Error("expected no-op on duplicate")
	}

	// Query pending.
	pending, err := db.PendingShards("test-ds", StageIngest)
	if err != nil {
		t.Fatal(err)
	}
	if len(pending) != 1 {
		t.Fatalf("expected 1 pending, got %d", len(pending))
	}
	if pending[0].ShardIdx != 0 || pending[0].Status != StatusPending {
		t.Errorf("unexpected shard: %+v", pending[0])
	}
}

func TestShardLifecycle(t *testing.T) {
	db := tempDB(t)

	db.EnsureShard("ds", 0, StageTokenize, "/in")

	// Mark running.
	if err := db.MarkRunning("ds", 0, StageTokenize); err != nil {
		t.Fatal(err)
	}

	shards, _ := db.ShardsByStatus("ds", StageTokenize, StatusRunning)
	if len(shards) != 1 {
		t.Fatalf("expected 1 running, got %d", len(shards))
	}

	// Mark done.
	if err := db.MarkDone("ds", 0, StageTokenize, "/out", 1000, 5, 4096); err != nil {
		t.Fatal(err)
	}

	shards, _ = db.ShardsByStatus("ds", StageTokenize, StatusDone)
	if len(shards) != 1 {
		t.Fatalf("expected 1 done, got %d", len(shards))
	}
	if shards[0].Tokens != 1000 || shards[0].Docs != 5 {
		t.Errorf("unexpected stats: tokens=%d docs=%d", shards[0].Tokens, shards[0].Docs)
	}
}

func TestResetInterrupted(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.db")

	// First open, insert a shard and mark it running.
	db, _ := Open(path)
	db.EnsureShard("ds", 0, StageProcess, "/in")
	db.MarkRunning("ds", 0, StageProcess)
	db.Close()

	// Re-open — should reset running → pending.
	db2, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()

	pending, _ := db2.PendingShards("ds", StageProcess)
	if len(pending) != 1 {
		t.Fatalf("expected interrupted shard to be reset to pending, got %d", len(pending))
	}
}

func TestMarkFailed(t *testing.T) {
	db := tempDB(t)

	db.EnsureShard("ds", 0, StageIngest, "/in")
	db.MarkRunning("ds", 0, StageIngest)
	db.MarkFailed("ds", 0, StageIngest, "disk full")

	shards, _ := db.ShardsByStatus("ds", StageIngest, StatusFailed)
	if len(shards) != 1 {
		t.Fatalf("expected 1 failed, got %d", len(shards))
	}
	if shards[0].Error != "disk full" {
		t.Errorf("unexpected error: %q", shards[0].Error)
	}

	// Pending should be empty — failed shards stay failed.
	pending, _ := db.PendingShards("ds", StageIngest)
	if len(pending) != 0 {
		t.Errorf("expected 0 pending, got %d", len(pending))
	}
}

func TestResetDataset(t *testing.T) {
	db := tempDB(t)

	db.EnsureShard("ds", 0, StageIngest, "/in")
	db.EnsureShard("ds", 1, StageIngest, "/in")
	db.EnsureShard("ds", 0, StageProcess, "/in")

	// Reset just ingest stage.
	n, err := db.ResetDataset("ds", StageIngest)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Errorf("expected 2 deleted, got %d", n)
	}

	// Process shard should still exist.
	shards, _ := db.AllShards("ds", StageProcess)
	if len(shards) != 1 {
		t.Errorf("expected 1 process shard remaining, got %d", len(shards))
	}
}

func TestResetFailed(t *testing.T) {
	db := tempDB(t)

	db.EnsureShard("ds", 0, StageIngest, "/in")
	db.EnsureShard("ds", 1, StageIngest, "/in")
	db.MarkRunning("ds", 0, StageIngest)
	db.MarkFailed("ds", 0, StageIngest, "err")
	db.MarkRunning("ds", 1, StageIngest)
	db.MarkDone("ds", 1, StageIngest, "/out", 0, 0, 0)

	n, err := db.ResetFailed("ds", StageIngest)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("expected 1 reset, got %d", n)
	}

	// Should now be pending.
	pending, _ := db.PendingShards("ds", StageIngest)
	if len(pending) != 1 {
		t.Errorf("expected 1 pending, got %d", len(pending))
	}
}

func TestStatusSummary(t *testing.T) {
	db := tempDB(t)

	db.EnsureShard("ds1", 0, StageIngest, "/in")
	db.EnsureShard("ds1", 1, StageIngest, "/in")
	db.MarkRunning("ds1", 0, StageIngest)
	db.MarkDone("ds1", 0, StageIngest, "/out", 100, 5, 1000)

	db.EnsureShard("ds2", 0, StageTokenize, "/in")

	summary, err := db.StatusSummary(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(summary) != 2 {
		t.Fatalf("expected 2 rows, got %d", len(summary))
	}

	// Filter to ds1.
	summary, _ = db.StatusSummary([]string{"ds1"})
	if len(summary) != 1 {
		t.Fatalf("expected 1 row, got %d", len(summary))
	}
	if summary[0].Done != 1 || summary[0].Pending != 1 {
		t.Errorf("unexpected: done=%d pending=%d", summary[0].Done, summary[0].Pending)
	}
}

func TestTotalTokens(t *testing.T) {
	db := tempDB(t)

	db.EnsureShard("ds", 0, StageTokenize, "/in")
	db.EnsureShard("ds", 1, StageTokenize, "/in")
	db.MarkRunning("ds", 0, StageTokenize)
	db.MarkDone("ds", 0, StageTokenize, "/out", 5000, 10, 0)
	db.MarkRunning("ds", 1, StageTokenize)
	db.MarkDone("ds", 1, StageTokenize, "/out", 3000, 6, 0)

	total, err := db.TotalTokens("ds", StageTokenize)
	if err != nil {
		t.Fatal(err)
	}
	if total != 8000 {
		t.Errorf("expected 8000 tokens, got %d", total)
	}
}

func TestStartFinishRun(t *testing.T) {
	db := tempDB(t)

	runID, err := db.StartRun("abc123", "run --config test.toml")
	if err != nil {
		t.Fatal(err)
	}
	if runID <= 0 {
		t.Errorf("expected positive run ID, got %d", runID)
	}

	if err := db.FinishRun(runID); err != nil {
		t.Fatal(err)
	}
}

func TestEnsureShardsBatch(t *testing.T) {
	db := tempDB(t)

	shards := []Shard{
		{Dataset: "ds", ShardIdx: 0, Stage: StageIngest, InputPath: "/a"},
		{Dataset: "ds", ShardIdx: 1, Stage: StageIngest, InputPath: "/b"},
		{Dataset: "ds", ShardIdx: 2, Stage: StageIngest, InputPath: "/c"},
	}

	if err := db.EnsureShardsBatch(shards); err != nil {
		t.Fatal(err)
	}

	all, _ := db.AllShards("ds", StageIngest)
	if len(all) != 3 {
		t.Fatalf("expected 3 shards, got %d", len(all))
	}

	// Re-insert should be idempotent.
	if err := db.EnsureShardsBatch(shards); err != nil {
		t.Fatal(err)
	}
	all, _ = db.AllShards("ds", StageIngest)
	if len(all) != 3 {
		t.Fatalf("expected still 3 shards, got %d", len(all))
	}
}

func init() {
	// Suppress state reset log output in tests.
	os.Setenv("DATAPREP_TEST", "1")
}
