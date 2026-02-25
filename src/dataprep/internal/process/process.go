package process

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/config"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/state"
)

// Processor runs Lua scripts over raw JSONL shards to produce processed JSONL.
type Processor struct {
	cfg        *config.Config
	db         *state.DB
	configPath string
}

// New creates a new Processor.
func New(cfg *config.Config, db *state.DB, configPath string) *Processor {
	return &Processor{cfg: cfg, db: db, configPath: configPath}
}

// Run processes raw JSONL shards for the given datasets.
func (p *Processor) Run(datasets []config.DatasetConfig) error {
	for _, ds := range datasets {
		if err := p.processDataset(ds); err != nil {
			return fmt.Errorf("process %s: %w", ds.Name, err)
		}
	}
	return nil
}

func (p *Processor) processDataset(ds config.DatasetConfig) error {
	rawDir := filepath.Join(p.cfg.Output.RawDir, ds.Name)
	outDir := filepath.Join(p.cfg.Output.ProcessedDir, ds.Name)

	// Find raw JSONL shards.
	entries, err := os.ReadDir(rawDir)
	if err != nil {
		return fmt.Errorf("read raw dir %s: %w", rawDir, err)
	}

	var rawFiles []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
			rawFiles = append(rawFiles, filepath.Join(rawDir, e.Name()))
		}
	}
	if len(rawFiles) == 0 {
		return fmt.Errorf("no raw JSONL shards found in %s (run ingest first?)", rawDir)
	}

	// Ensure output directory exists.
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	// Register shards in state DB.
	for i, rawPath := range rawFiles {
		p.db.EnsureShard(ds.Name, i, state.StageProcess, rawPath)
	}

	// Get pending shards.
	pending, err := p.db.PendingShards(ds.Name, state.StageProcess)
	if err != nil {
		return err
	}

	if len(pending) == 0 {
		log.Printf("[process] %s: all %d shards already done, skipping", ds.Name, len(rawFiles))
		return nil
	}

	log.Printf("[process] %s: %d/%d shards to process (script: %s)", ds.Name, len(pending), len(rawFiles), ds.Script)

	// Resolve script path.
	scriptPath := p.cfg.ResolveScript(p.configPath, ds.Script)

	for _, sh := range pending {
		if err := p.processShard(ds, sh, outDir, scriptPath); err != nil {
			log.Printf("[process] %s shard %d: ERROR %v", ds.Name, sh.ShardIdx, err)
		}
	}

	return nil
}

func (p *Processor) processShard(ds config.DatasetConfig, sh state.Shard, outDir, scriptPath string) error {
	if err := p.db.MarkRunning(ds.Name, sh.ShardIdx, state.StageProcess); err != nil {
		return err
	}

	outPath := filepath.Join(outDir, fmt.Sprintf("shard_%05d.jsonl", sh.ShardIdx))

	// Create Lua environment for this shard.
	env, err := NewLuaEnv(scriptPath)
	if err != nil {
		p.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageProcess, err.Error())
		return err
	}
	defer env.Close()

	// Open input.
	inFile, err := os.Open(sh.InputPath)
	if err != nil {
		p.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageProcess, err.Error())
		return err
	}
	defer inFile.Close()

	// Open output.
	outFile, err := os.Create(outPath)
	if err != nil {
		p.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageProcess, err.Error())
		return err
	}
	bw := bufio.NewWriterSize(outFile, 1<<20)
	enc := json.NewEncoder(bw)
	enc.SetEscapeHTML(false)

	scanner := bufio.NewScanner(inFile)
	scanner.Buffer(make([]byte, 0), 10*1024*1024) // 10MB max line

	var docs, skipped int64
	var totalBytes int64

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// Parse the raw JSONL record.
		var record map[string]any
		if err := json.Unmarshal(line, &record); err != nil {
			skipped++
			continue
		}

		// Run through Lua.
		result, err := env.Extract(record)
		if err != nil {
			log.Printf("[process] %s shard %d: lua error: %v", ds.Name, sh.ShardIdx, err)
			skipped++
			continue
		}
		if result == nil {
			skipped++ // Lua returned nil = skip
			continue
		}
		if result.Route == "skip" {
			skipped++
			continue
		}
		if result.PTText == "" {
			skipped++
			continue
		}

		if err := enc.Encode(result); err != nil {
			bw.Flush()
			outFile.Close()
			os.Remove(outPath)
			p.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageProcess, err.Error())
			return err
		}
		docs++
	}

	if err := scanner.Err(); err != nil {
		bw.Flush()
		outFile.Close()
		os.Remove(outPath)
		p.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageProcess, err.Error())
		return err
	}

	if err := bw.Flush(); err != nil {
		outFile.Close()
		os.Remove(outPath)
		p.db.MarkFailed(ds.Name, sh.ShardIdx, state.StageProcess, err.Error())
		return err
	}

	stat, _ := outFile.Stat()
	if stat != nil {
		totalBytes = stat.Size()
	}
	outFile.Close()

	log.Printf("[process] %s shard %d: %d docs, %d skipped, %.1f MB",
		ds.Name, sh.ShardIdx, docs, skipped, float64(totalBytes)/(1<<20))

	return p.db.MarkDone(ds.Name, sh.ShardIdx, state.StageProcess, outPath, 0, docs, totalBytes)
}
