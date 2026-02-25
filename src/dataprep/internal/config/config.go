// Package config loads pipeline.toml and provides typed access to all settings.
package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

// Config is the top-level pipeline configuration.
type Config struct {
	Tokenizer TokenizerConfig `toml:"tokenizer"`
	Output    OutputConfig    `toml:"output"`
	Defaults  DefaultsConfig  `toml:"defaults"`
	Datasets  []DatasetConfig `toml:"dataset"`
}

// TokenizerConfig points to the tokenizer artifacts.
type TokenizerConfig struct {
	Path       string `toml:"path"`
	EOSTokenID int    `toml:"eos_token_id"`
}

// OutputConfig defines directory layout for all pipeline outputs.
type OutputConfig struct {
	BaseDir      string `toml:"base_dir"`
	TrainDir     string `toml:"train_dir"`
	ValDir       string `toml:"val_dir"`
	SFTDir       string `toml:"sft_dir"`
	RawDir       string `toml:"raw_dir"`
	ProcessedDir string `toml:"processed_dir"`
	StateDB      string `toml:"state_db"`
	HFCache      string `toml:"hf_cache"`
}

// DefaultsConfig provides fallback values for datasets that don't override them.
type DefaultsConfig struct {
	TargetTokens int64   `toml:"target_tokens"`
	ValFraction  float64 `toml:"val_fraction"`
	ShardSize    int64   `toml:"shard_size"`
	Workers      int     `toml:"workers"`
	Script       string  `toml:"script"`
}

// DatasetConfig defines a single dataset in the pipeline.
type DatasetConfig struct {
	Name         string  `toml:"name"`
	HFPath       string  `toml:"hf_path"`
	HFSubset     string  `toml:"hf_subset"`
	TextColumn   string  `toml:"text_column"`
	Category     string  `toml:"category"`
	TargetTokens int64   `toml:"target_tokens"`
	ValFraction  float64 `toml:"val_fraction"`
	ShardSize    int64   `toml:"shard_size"`
	Script       string  `toml:"script"`
	SFTCapable   bool    `toml:"sft_capable"`
	Include      string  `toml:"include"` // glob for filtering parquet files (e.g. "*.parquet")
}

// Load reads a TOML config file and applies defaults.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := toml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("parse config %s: %w", path, err)
	}

	cfg.applyDefaults()

	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &cfg, nil
}

func (c *Config) applyDefaults() {
	// Output defaults
	if c.Output.BaseDir == "" {
		c.Output.BaseDir = "tokenized"
	}
	if c.Output.TrainDir == "" {
		c.Output.TrainDir = filepath.Join(c.Output.BaseDir, "train")
	}
	if c.Output.ValDir == "" {
		c.Output.ValDir = filepath.Join(c.Output.BaseDir, "val")
	}
	if c.Output.SFTDir == "" {
		c.Output.SFTDir = filepath.Join(c.Output.BaseDir, "sft")
	}
	if c.Output.RawDir == "" {
		c.Output.RawDir = ".dataprep/raw"
	}
	if c.Output.ProcessedDir == "" {
		c.Output.ProcessedDir = ".dataprep/processed"
	}
	if c.Output.StateDB == "" {
		c.Output.StateDB = ".dataprep/state.db"
	}
	if c.Output.HFCache == "" {
		c.Output.HFCache = ".dataprep/hf_cache"
	}

	// Global defaults
	if c.Defaults.TargetTokens == 0 {
		c.Defaults.TargetTokens = 1_000_000_000
	}
	if c.Defaults.ValFraction == 0 {
		c.Defaults.ValFraction = 0.005
	}
	if c.Defaults.ShardSize == 0 {
		c.Defaults.ShardSize = 67_108_864 // 64 MB
	}
	if c.Defaults.Workers == 0 {
		c.Defaults.Workers = 16
	}
	if c.Defaults.Script == "" {
		c.Defaults.Script = "scripts/_default.lua"
	}

	// Per-dataset defaults
	for i := range c.Datasets {
		ds := &c.Datasets[i]
		if ds.TargetTokens == 0 {
			ds.TargetTokens = c.Defaults.TargetTokens
		}
		if ds.ValFraction == 0 {
			ds.ValFraction = c.Defaults.ValFraction
		}
		if ds.ShardSize == 0 {
			ds.ShardSize = c.Defaults.ShardSize
		}
		if ds.Script == "" {
			ds.Script = c.Defaults.Script
		}
		if ds.TextColumn == "" {
			ds.TextColumn = "text"
		}
	}
}

func (c *Config) validate() error {
	if c.Tokenizer.Path == "" {
		return fmt.Errorf("tokenizer.path is required")
	}

	seen := make(map[string]bool)
	for i, ds := range c.Datasets {
		if ds.Name == "" {
			return fmt.Errorf("dataset[%d]: name is required", i)
		}
		if seen[ds.Name] {
			return fmt.Errorf("dataset[%d]: duplicate name %q", i, ds.Name)
		}
		seen[ds.Name] = true

		if ds.HFPath == "" {
			return fmt.Errorf("dataset %q: hf_path is required", ds.Name)
		}
		if ds.Category == "" {
			return fmt.Errorf("dataset %q: category is required", ds.Name)
		}
	}
	return nil
}

// DatasetByName returns the config for a named dataset, or nil if not found.
func (c *Config) DatasetByName(name string) *DatasetConfig {
	for i := range c.Datasets {
		if c.Datasets[i].Name == name {
			return &c.Datasets[i]
		}
	}
	return nil
}

// DatasetNames returns all dataset names in config order.
func (c *Config) DatasetNames() []string {
	names := make([]string, len(c.Datasets))
	for i, ds := range c.Datasets {
		names[i] = ds.Name
	}
	return names
}

// FilterDatasets returns only the named datasets. If names is empty, returns all.
func (c *Config) FilterDatasets(names []string) []DatasetConfig {
	if len(names) == 0 {
		return c.Datasets
	}
	want := make(map[string]bool, len(names))
	for _, n := range names {
		want[n] = true
	}
	var result []DatasetConfig
	for _, ds := range c.Datasets {
		if want[ds.Name] {
			result = append(result, ds)
		}
	}
	return result
}

// ScriptDir returns the directory containing Lua scripts, resolved relative to the config file.
func (c *Config) ScriptDir(configPath string) string {
	return filepath.Join(filepath.Dir(configPath), "scripts")
}

// ResolveScript resolves a script path relative to the scripts directory.
func (c *Config) ResolveScript(configPath string, scriptRef string) string {
	if filepath.IsAbs(scriptRef) {
		return scriptRef
	}
	return filepath.Join(filepath.Dir(configPath), scriptRef)
}
