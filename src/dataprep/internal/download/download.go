// Package download shells out to huggingface-cli to download dataset parquet files.
package download

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/config"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/state"
)

// Downloader manages HuggingFace dataset downloads.
type Downloader struct {
	cfg *config.Config
	db  *state.DB
}

// New creates a new Downloader.
func New(cfg *config.Config, db *state.DB) *Downloader {
	return &Downloader{cfg: cfg, db: db}
}

// Run downloads parquet files for the given datasets.
// If datasets is empty, downloads all configured datasets.
func (d *Downloader) Run(datasets []config.DatasetConfig) error {
	for _, ds := range datasets {
		if err := d.downloadDataset(ds); err != nil {
			return fmt.Errorf("download %s: %w", ds.Name, err)
		}
	}
	return nil
}

func (d *Downloader) downloadDataset(ds config.DatasetConfig) error {
	destDir := filepath.Join(d.cfg.Output.HFCache, ds.Name)

	// Check if already downloaded by looking for parquet files.
	if isDownloaded(destDir) {
		log.Printf("[download] %s: already downloaded to %s, skipping", ds.Name, destDir)
		// Ensure state is recorded.
		d.db.EnsureShard(ds.Name, 0, state.StageDownload, "")
		d.db.MarkDone(ds.Name, 0, state.StageDownload, destDir, 0, 0, 0)
		return nil
	}

	// Register in state.
	d.db.EnsureShard(ds.Name, 0, state.StageDownload, "")
	if err := d.db.MarkRunning(ds.Name, 0, state.StageDownload); err != nil {
		return err
	}

	log.Printf("[download] %s: downloading %s to %s", ds.Name, ds.HFPath, destDir)

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		d.db.MarkFailed(ds.Name, 0, state.StageDownload, err.Error())
		return err
	}

	args := []string{
		"download",
		ds.HFPath,
		"--repo-type", "dataset",
		"--local-dir", destDir,
	}

	// If there's a subset, we include only that subset's parquet files.
	if ds.HFSubset != "" {
		args = append(args, "--include", fmt.Sprintf("%s/*.parquet", ds.HFSubset))
	} else {
		args = append(args, "--include", "*.parquet")
	}

	// If there's an include pattern override, use that instead.
	if ds.Include != "" {
		// Replace the last --include with the custom one.
		args[len(args)-1] = ds.Include
	}

	cmd := exec.Command("huggingface-cli", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		errMsg := fmt.Sprintf("huggingface-cli download failed: %v", err)
		d.db.MarkFailed(ds.Name, 0, state.StageDownload, errMsg)
		return fmt.Errorf("%s", errMsg)
	}

	// Verify we got some parquet files.
	parquets, err := findParquetFiles(destDir)
	if err != nil {
		errMsg := fmt.Sprintf("scan downloaded files: %v", err)
		d.db.MarkFailed(ds.Name, 0, state.StageDownload, errMsg)
		return fmt.Errorf("%s", errMsg)
	}
	if len(parquets) == 0 {
		errMsg := "no parquet files found after download"
		d.db.MarkFailed(ds.Name, 0, state.StageDownload, errMsg)
		return fmt.Errorf("%s: %s", ds.Name, errMsg)
	}

	// Compute total bytes downloaded.
	var totalBytes int64
	for _, p := range parquets {
		info, err := os.Stat(p)
		if err == nil {
			totalBytes += info.Size()
		}
	}

	log.Printf("[download] %s: downloaded %d parquet files (%.1f GB)",
		ds.Name, len(parquets), float64(totalBytes)/(1<<30))

	return d.db.MarkDone(ds.Name, 0, state.StageDownload, destDir, 0, 0, totalBytes)
}

// isDownloaded checks if a directory contains at least one parquet file.
func isDownloaded(dir string) bool {
	files, err := findParquetFiles(dir)
	return err == nil && len(files) > 0
}

// findParquetFiles recursively finds all .parquet files in a directory.
func findParquetFiles(dir string) ([]string, error) {
	var files []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip errors
		}
		if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".parquet") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

// FindParquetFiles is the exported version for use by other packages.
func FindParquetFiles(dir string) ([]string, error) {
	return findParquetFiles(dir)
}
