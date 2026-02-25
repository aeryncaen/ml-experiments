// Package pipeline chains download → ingest → process → tokenize stages.
package pipeline

import (
	"crypto/sha256"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/config"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/download"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/ingest"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/process"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/state"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/tokenize"
)

// Pipeline orchestrates the full dataprep workflow.
type Pipeline struct {
	cfg        *config.Config
	db         *state.DB
	configPath string
}

// New creates a new Pipeline.
func New(cfg *config.Config, db *state.DB, configPath string) *Pipeline {
	return &Pipeline{cfg: cfg, db: db, configPath: configPath}
}

// Run executes the given stages for the specified datasets.
// If datasets is empty, processes all configured datasets.
func (p *Pipeline) Run(stages []state.Stage, datasetNames []string) error {
	datasets := p.cfg.FilterDatasets(datasetNames)
	if len(datasets) == 0 {
		return fmt.Errorf("no datasets to process")
	}

	// Ensure output directories exist.
	for _, dir := range []string{
		p.cfg.Output.RawDir,
		p.cfg.Output.ProcessedDir,
		p.cfg.Output.TrainDir,
		p.cfg.Output.ValDir,
		p.cfg.Output.SFTDir,
		p.cfg.Output.HFCache,
	} {
		os.MkdirAll(dir, 0o755)
	}

	// Record the run.
	configHash := hashConfig(p.configPath)
	stageNames := make([]string, len(stages))
	for i, s := range stages {
		stageNames[i] = string(s)
	}
	dsNames := make([]string, len(datasets))
	for i, ds := range datasets {
		dsNames[i] = ds.Name
	}
	command := fmt.Sprintf("stages=[%s] datasets=[%s]", strings.Join(stageNames, ","), strings.Join(dsNames, ","))
	runID, _ := p.db.StartRun(configHash, command)

	startTime := time.Now()
	log.Printf("═══════════════════════════════════════════════════")
	log.Printf("Pipeline starting: %s", command)
	log.Printf("═══════════════════════════════════════════════════")

	var pipelineErr error

	for _, stage := range stages {
		stageStart := time.Now()
		log.Printf("")
		log.Printf("─── Stage: %s ────────────────────────────────", stage)

		var err error
		switch stage {
		case state.StageDownload:
			err = p.runDownload(datasets)
		case state.StageIngest:
			err = p.runIngest(datasets)
		case state.StageProcess:
			err = p.runProcess(datasets)
		case state.StageTokenize:
			err = p.runTokenize(datasets)
		default:
			err = fmt.Errorf("unknown stage: %s", stage)
		}

		elapsed := time.Since(stageStart)
		if err != nil {
			log.Printf("─── Stage %s FAILED after %v: %v", stage, elapsed.Round(time.Second), err)
			pipelineErr = err
			break
		}
		log.Printf("─── Stage %s completed in %v", stage, elapsed.Round(time.Second))
	}

	elapsed := time.Since(startTime)
	if runID > 0 {
		p.db.FinishRun(runID)
	}

	log.Printf("")
	log.Printf("═══════════════════════════════════════════════════")
	if pipelineErr != nil {
		log.Printf("Pipeline FAILED after %v: %v", elapsed.Round(time.Second), pipelineErr)
	} else {
		log.Printf("Pipeline completed in %v", elapsed.Round(time.Second))
	}

	// Print token totals.
	for _, ds := range datasets {
		tokens, err := p.db.TotalTokens(ds.Name, state.StageTokenize)
		if err == nil && tokens > 0 {
			log.Printf("  %-28s %s tokens", ds.Name, formatTokens(tokens))
		}
	}
	log.Printf("═══════════════════════════════════════════════════")

	return pipelineErr
}

func (p *Pipeline) runDownload(datasets []config.DatasetConfig) error {
	dl := download.New(p.cfg, p.db)
	return dl.Run(datasets)
}

func (p *Pipeline) runIngest(datasets []config.DatasetConfig) error {
	ing := ingest.New(p.cfg, p.db, p.configPath)
	return ing.Run(datasets)
}

func (p *Pipeline) runProcess(datasets []config.DatasetConfig) error {
	proc := process.New(p.cfg, p.db, p.configPath)
	return proc.Run(datasets)
}

func (p *Pipeline) runTokenize(datasets []config.DatasetConfig) error {
	tok := tokenize.New(p.cfg, p.db, p.configPath)
	return tok.Run(datasets)
}

func hashConfig(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return "unknown"
	}
	h := sha256.Sum256(data)
	return fmt.Sprintf("%x", h[:8])
}

func formatTokens(n int64) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}
