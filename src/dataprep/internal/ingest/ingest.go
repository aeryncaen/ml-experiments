package ingest

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/config"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/download"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/state"
)

// Ingestor converts downloaded parquet files into raw JSONL shards.
type Ingestor struct {
	cfg        *config.Config
	db         *state.DB
	configPath string
}

// New creates a new Ingestor.
func New(cfg *config.Config, db *state.DB, configPath string) *Ingestor {
	return &Ingestor{cfg: cfg, db: db, configPath: configPath}
}

// Run ingests parquet files for the given datasets into JSONL shards.
func (ing *Ingestor) Run(datasets []config.DatasetConfig) error {
	for _, ds := range datasets {
		if err := ing.ingestDataset(ds); err != nil {
			return fmt.Errorf("ingest %s: %w", ds.Name, err)
		}
	}
	return nil
}

func (ing *Ingestor) ingestDataset(ds config.DatasetConfig) error {
	cacheDir := filepath.Join(ing.cfg.Output.HFCache, ds.Name)
	outDir := filepath.Join(ing.cfg.Output.RawDir, ds.Name)

	// Find parquet files.
	parquets, err := download.FindParquetFiles(cacheDir)
	if err != nil {
		return fmt.Errorf("find parquet files in %s: %w", cacheDir, err)
	}
	if len(parquets) == 0 {
		return fmt.Errorf("no parquet files found in %s (run download first?)", cacheDir)
	}
	sort.Strings(parquets)

	// Ensure output directory exists.
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	// Register shards in state DB.
	for i, pqPath := range parquets {
		ing.db.EnsureShard(ds.Name, i, state.StageIngest, pqPath)
	}

	// Get pending shards.
	pending, err := ing.db.PendingShards(ds.Name, state.StageIngest)
	if err != nil {
		return err
	}

	if len(pending) == 0 {
		log.Printf("[ingest] %s: all %d shards already done, skipping", ds.Name, len(parquets))
		return nil
	}

	log.Printf("[ingest] %s: %d/%d shards to process", ds.Name, len(pending), len(parquets))

	for _, sh := range pending {
		if err := ing.ingestShard(ds, sh, outDir); err != nil {
			log.Printf("[ingest] %s shard %d: ERROR %v", ds.Name, sh.ShardIdx, err)
			// Continue with other shards.
		}
	}

	return nil
}

func (ing *Ingestor) ingestShard(ds config.DatasetConfig, sh state.Shard, outDir string) error {
	if err := ing.db.MarkRunning(ds.Name, sh.ShardIdx, state.StageIngest); err != nil {
		return err
	}

	outPath := filepath.Join(outDir, fmt.Sprintf("shard_%05d.jsonl", sh.ShardIdx))

	// Open parquet file.
	pqReader, err := OpenParquet(sh.InputPath)
	if err != nil {
		ing.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageIngest, err.Error())
		return err
	}
	defer pqReader.Close()

	// Open output JSONL file.
	outFile, err := os.Create(outPath)
	if err != nil {
		ing.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageIngest, err.Error())
		return err
	}
	bw := bufio.NewWriterSize(outFile, 1<<20)
	enc := json.NewEncoder(bw)
	enc.SetEscapeHTML(false)

	var docs int64
	var bytes int64

	if err := pqReader.ReadAll(func(record map[string]any) error {
		if err := enc.Encode(record); err != nil {
			return err
		}
		docs++
		return nil
	}); err != nil {
		bw.Flush()
		outFile.Close()
		os.Remove(outPath)
		ing.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageIngest, err.Error())
		return err
	}

	if err := bw.Flush(); err != nil {
		outFile.Close()
		os.Remove(outPath)
		ing.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageIngest, err.Error())
		return err
	}

	stat, _ := outFile.Stat()
	if stat != nil {
		bytes = stat.Size()
	}
	outFile.Close()

	log.Printf("[ingest] %s shard %d: %d docs, %.1f MB",
		ds.Name, sh.ShardIdx, docs, float64(bytes)/(1<<20))

	return ing.db.MarkDone(ds.Name, sh.ShardIdx, state.StageIngest, outPath, 0, docs, bytes)
}
