// Command dataprep processes HuggingFace datasets into training-ready shards.
//
// Usage:
//
//	dataprep run      --config pipeline.toml [--datasets name1,name2]
//	dataprep download --config pipeline.toml [--datasets name1,name2]
//	dataprep ingest   --config pipeline.toml [--datasets name1,name2]
//	dataprep process  --config pipeline.toml [--datasets name1,name2]
//	dataprep tokenize --config pipeline.toml [--datasets name1,name2]
//	dataprep status   --config pipeline.toml [--datasets name1,name2]
//	dataprep reset    --config pipeline.toml --datasets name1 [--stage download]
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/config"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/pipeline"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/state"
)

func main() {
	log.SetFlags(log.Ltime | log.Lmsgprefix)

	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	subcmd := os.Args[1]
	fs := flag.NewFlagSet(subcmd, flag.ExitOnError)
	configPath := fs.String("config", "pipeline.toml", "Path to pipeline.toml config")
	datasetsFlag := fs.String("datasets", "", "Comma-separated dataset names (empty = all)")
	stageFlag := fs.String("stage", "", "Stage to reset (for reset command)")
	workersFlag := fs.Int("workers", 0, "Override number of workers (0 = use config default)")
	fs.Parse(os.Args[2:])

	var datasetNames []string
	if *datasetsFlag != "" {
		datasetNames = strings.Split(*datasetsFlag, ",")
		for i := range datasetNames {
			datasetNames[i] = strings.TrimSpace(datasetNames[i])
		}
	}

	switch subcmd {
	case "run", "download", "ingest", "process", "tokenize":
		runPipeline(subcmd, *configPath, datasetNames, *workersFlag)
	case "status":
		runStatus(*configPath, datasetNames)
	case "reset":
		runReset(*configPath, datasetNames, *stageFlag)
	case "help", "-h", "--help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", subcmd)
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: dataprep <command> [flags]

Commands:
  run       Full pipeline: download → ingest → process → tokenize
  download  Download parquet from HuggingFace
  ingest    Convert parquet → raw JSONL shards
  process   Run Lua scripts on raw JSONL → processed JSONL
  tokenize  Tokenize processed JSONL → .bin/.idx/.mask shards
  status    Print per-dataset progress
  reset     Reset a dataset or stage to re-run it

Flags:
  --config   Path to pipeline.toml (default: pipeline.toml)
  --datasets Comma-separated dataset names (default: all)
  --stage    Stage to reset (for reset command)
  --workers  Override number of workers
`)
}

func loadConfigAndState(configPath string) (*config.Config, *state.DB) {
	cfg, err := config.Load(configPath)
	if err != nil {
		log.Fatalf("load config: %v", err)
	}

	// Ensure state directory exists.
	if err := os.MkdirAll(strings.TrimSuffix(cfg.Output.StateDB, "/state.db"), 0o755); err != nil {
		log.Fatalf("create state dir: %v", err)
	}

	db, err := state.Open(cfg.Output.StateDB)
	if err != nil {
		log.Fatalf("open state db: %v", err)
	}

	return cfg, db
}

func runPipeline(subcmd string, configPath string, datasets []string, workers int) {
	cfg, db := loadConfigAndState(configPath)
	defer db.Close()

	if workers > 0 {
		cfg.Defaults.Workers = workers
	}

	// Validate requested datasets exist in config.
	if len(datasets) > 0 {
		for _, name := range datasets {
			if cfg.DatasetByName(name) == nil {
				log.Fatalf("unknown dataset: %q (not in config)", name)
			}
		}
	}

	// Map subcommand to stages to run.
	var stages []state.Stage
	switch subcmd {
	case "run":
		stages = []state.Stage{state.StageDownload, state.StageIngest, state.StageProcess, state.StageTokenize}
	case "download":
		stages = []state.Stage{state.StageDownload}
	case "ingest":
		stages = []state.Stage{state.StageIngest}
	case "process":
		stages = []state.Stage{state.StageProcess}
	case "tokenize":
		stages = []state.Stage{state.StageTokenize}
	}

	p := pipeline.New(cfg, db, configPath)
	if err := p.Run(stages, datasets); err != nil {
		log.Fatalf("pipeline failed: %v", err)
	}
}

func runStatus(configPath string, datasets []string) {
	cfg, db := loadConfigAndState(configPath)
	defer db.Close()

	if len(datasets) == 0 {
		datasets = cfg.DatasetNames()
	}

	summary, err := db.StatusSummary(datasets)
	if err != nil {
		log.Fatalf("query status: %v", err)
	}

	// Group by dataset.
	type dsRow struct {
		name   string
		stages map[state.Stage]state.DatasetStageStatus
		tokens int64
	}

	byDS := make(map[string]*dsRow)
	var order []string
	for _, s := range summary {
		r, ok := byDS[s.Dataset]
		if !ok {
			r = &dsRow{name: s.Dataset, stages: make(map[state.Stage]state.DatasetStageStatus)}
			byDS[s.Dataset] = r
			order = append(order, s.Dataset)
		}
		r.stages[s.Stage] = s
		if s.Stage == state.StageTokenize {
			r.tokens = s.Tokens
		}
	}

	stageOrder := []state.Stage{state.StageDownload, state.StageIngest, state.StageProcess, state.StageTokenize}

	fmt.Printf("\n%-28s", "Dataset")
	for _, st := range stageOrder {
		fmt.Printf(" %10s", st)
	}
	fmt.Printf(" %10s\n", "Tokens")
	fmt.Println(strings.Repeat("─", 82))

	for _, name := range order {
		r := byDS[name]
		fmt.Printf("%-28s", r.name)
		for _, st := range stageOrder {
			s, ok := r.stages[st]
			if !ok {
				fmt.Printf(" %10s", "·")
			} else {
				icon := "·"
				switch {
				case s.Failed > 0:
					icon = "✗"
				case s.Done == s.Total:
					icon = "✓"
				case s.Running > 0 || s.Done > 0:
					icon = "●"
				}
				fmt.Printf(" %s %d/%d", icon, s.Done, s.Total)
			}
		}
		if r.tokens > 0 {
			fmt.Printf(" %9s", formatTokens(r.tokens))
		} else {
			fmt.Printf(" %10s", "—")
		}
		fmt.Println()
	}

	fmt.Println(strings.Repeat("─", 82))
	fmt.Println("✓ = done   ● = in progress   · = pending   ✗ = failed")
}

func runReset(configPath string, datasets []string, stageStr string) {
	_, db := loadConfigAndState(configPath)
	defer db.Close()

	if len(datasets) == 0 {
		log.Fatal("reset requires --datasets")
	}

	var stage state.Stage
	if stageStr != "" {
		stage = state.Stage(stageStr)
	}

	for _, ds := range datasets {
		n, err := db.ResetDataset(ds, stage)
		if err != nil {
			log.Fatalf("reset %s: %v", ds, err)
		}
		if stage == "" {
			fmt.Printf("reset %s: deleted %d shard records (all stages)\n", ds, n)
		} else {
			fmt.Printf("reset %s/%s: deleted %d shard records\n", ds, stage, n)
		}
	}
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
