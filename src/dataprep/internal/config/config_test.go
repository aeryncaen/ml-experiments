package config

import (
	"os"
	"path/filepath"
	"testing"
)

const minimalTOML = `
[tokenizer]
path = "/tmp/tokenizer"
eos_token_id = 42

[[dataset]]
name = "test-ds"
hf_path = "org/repo"
category = "web"
`

const fullTOML = `
[tokenizer]
path = "/tmp/tokenizer"
eos_token_id = 149727

[output]
base_dir = "/data/out"
train_dir = "/data/out/train"
val_dir = "/data/out/val"
sft_dir = "/data/out/sft"
raw_dir = "/data/raw"
processed_dir = "/data/processed"
state_db = "/data/state.db"
hf_cache = "/data/hf_cache"

[defaults]
target_tokens = 500_000_000
val_fraction = 0.01
shard_size = 33_554_432
workers = 8
script = "scripts/custom.lua"

[[dataset]]
name = "ds-alpha"
hf_path = "org/alpha"
category = "web"
target_tokens = 1_000_000
val_fraction = 0.05
shard_size = 1_048_576
script = "scripts/alpha.lua"
text_column = "content"
sft_capable = true

[[dataset]]
name = "ds-beta"
hf_path = "org/beta"
hf_subset = "en"
category = "code"
include = "*.parquet"
`

func writeTempTOML(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "pipeline.toml")
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadMinimal(t *testing.T) {
	cfg, err := Load(writeTempTOML(t, minimalTOML))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Tokenizer.Path != "/tmp/tokenizer" {
		t.Errorf("tokenizer.path = %q", cfg.Tokenizer.Path)
	}
	if cfg.Tokenizer.EOSTokenID != 42 {
		t.Errorf("eos_token_id = %d", cfg.Tokenizer.EOSTokenID)
	}
	if len(cfg.Datasets) != 1 {
		t.Fatalf("datasets len = %d", len(cfg.Datasets))
	}
	ds := cfg.Datasets[0]
	if ds.Name != "test-ds" {
		t.Errorf("name = %q", ds.Name)
	}
	if ds.TextColumn != "text" {
		t.Errorf("text_column default = %q, want 'text'", ds.TextColumn)
	}
}

func TestDefaults(t *testing.T) {
	cfg, err := Load(writeTempTOML(t, minimalTOML))
	if err != nil {
		t.Fatal(err)
	}
	// Output defaults
	if cfg.Output.BaseDir != "tokenized" {
		t.Errorf("base_dir = %q", cfg.Output.BaseDir)
	}
	if cfg.Output.TrainDir != "tokenized/train" {
		t.Errorf("train_dir = %q", cfg.Output.TrainDir)
	}
	if cfg.Output.StateDB != ".dataprep/state.db" {
		t.Errorf("state_db = %q", cfg.Output.StateDB)
	}
	// Global defaults
	if cfg.Defaults.TargetTokens != 1_000_000_000 {
		t.Errorf("default target_tokens = %d", cfg.Defaults.TargetTokens)
	}
	if cfg.Defaults.ValFraction != 0.005 {
		t.Errorf("default val_fraction = %f", cfg.Defaults.ValFraction)
	}
	if cfg.Defaults.ShardSize != 67_108_864 {
		t.Errorf("default shard_size = %d", cfg.Defaults.ShardSize)
	}
	if cfg.Defaults.Workers != 16 {
		t.Errorf("default workers = %d", cfg.Defaults.Workers)
	}
	if cfg.Defaults.Script != "scripts/_default.lua" {
		t.Errorf("default script = %q", cfg.Defaults.Script)
	}
	// Dataset inherits defaults
	ds := cfg.Datasets[0]
	if ds.TargetTokens != 1_000_000_000 {
		t.Errorf("ds target_tokens = %d", ds.TargetTokens)
	}
	if ds.Script != "scripts/_default.lua" {
		t.Errorf("ds script = %q", ds.Script)
	}
}

func TestFullConfig(t *testing.T) {
	cfg, err := Load(writeTempTOML(t, fullTOML))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Datasets) != 2 {
		t.Fatalf("datasets len = %d", len(cfg.Datasets))
	}
	// ds-alpha overrides
	alpha := cfg.Datasets[0]
	if alpha.TargetTokens != 1_000_000 {
		t.Errorf("alpha target_tokens = %d", alpha.TargetTokens)
	}
	if alpha.ValFraction != 0.05 {
		t.Errorf("alpha val_fraction = %f", alpha.ValFraction)
	}
	if alpha.Script != "scripts/alpha.lua" {
		t.Errorf("alpha script = %q", alpha.Script)
	}
	if alpha.TextColumn != "content" {
		t.Errorf("alpha text_column = %q", alpha.TextColumn)
	}
	if !alpha.SFTCapable {
		t.Error("alpha should be sft_capable")
	}
	// ds-beta inherits defaults
	beta := cfg.Datasets[1]
	if beta.TargetTokens != 500_000_000 {
		t.Errorf("beta target_tokens = %d", beta.TargetTokens)
	}
	if beta.Script != "scripts/custom.lua" {
		t.Errorf("beta script = %q", beta.Script)
	}
	if beta.HFSubset != "en" {
		t.Errorf("beta hf_subset = %q", beta.HFSubset)
	}
	if beta.Include != "*.parquet" {
		t.Errorf("beta include = %q", beta.Include)
	}
}

func TestDatasetByName(t *testing.T) {
	cfg, err := Load(writeTempTOML(t, fullTOML))
	if err != nil {
		t.Fatal(err)
	}
	ds := cfg.DatasetByName("ds-alpha")
	if ds == nil || ds.Name != "ds-alpha" {
		t.Error("DatasetByName(ds-alpha) failed")
	}
	if cfg.DatasetByName("nonexistent") != nil {
		t.Error("DatasetByName(nonexistent) should return nil")
	}
}

func TestDatasetNames(t *testing.T) {
	cfg, err := Load(writeTempTOML(t, fullTOML))
	if err != nil {
		t.Fatal(err)
	}
	names := cfg.DatasetNames()
	if len(names) != 2 || names[0] != "ds-alpha" || names[1] != "ds-beta" {
		t.Errorf("DatasetNames() = %v", names)
	}
}

func TestFilterDatasets(t *testing.T) {
	cfg, err := Load(writeTempTOML(t, fullTOML))
	if err != nil {
		t.Fatal(err)
	}
	// Filter to one
	filtered := cfg.FilterDatasets([]string{"ds-beta"})
	if len(filtered) != 1 || filtered[0].Name != "ds-beta" {
		t.Errorf("FilterDatasets([ds-beta]) = %v", filtered)
	}
	// Empty = all
	all := cfg.FilterDatasets(nil)
	if len(all) != 2 {
		t.Errorf("FilterDatasets(nil) len = %d", len(all))
	}
}

func TestValidation_MissingTokenizerPath(t *testing.T) {
	toml := `
[tokenizer]
eos_token_id = 42
[[dataset]]
name = "x"
hf_path = "org/repo"
category = "web"
`
	_, err := Load(writeTempTOML(t, toml))
	if err == nil {
		t.Fatal("expected error for missing tokenizer.path")
	}
}

func TestValidation_MissingName(t *testing.T) {
	toml := `
[tokenizer]
path = "/tmp/tok"
[[dataset]]
hf_path = "org/repo"
category = "web"
`
	_, err := Load(writeTempTOML(t, toml))
	if err == nil {
		t.Fatal("expected error for missing dataset name")
	}
}

func TestValidation_DuplicateName(t *testing.T) {
	toml := `
[tokenizer]
path = "/tmp/tok"
[[dataset]]
name = "dup"
hf_path = "org/repo"
category = "web"
[[dataset]]
name = "dup"
hf_path = "org/repo2"
category = "code"
`
	_, err := Load(writeTempTOML(t, toml))
	if err == nil {
		t.Fatal("expected error for duplicate dataset name")
	}
}

func TestValidation_MissingHFPath(t *testing.T) {
	toml := `
[tokenizer]
path = "/tmp/tok"
[[dataset]]
name = "x"
category = "web"
`
	_, err := Load(writeTempTOML(t, toml))
	if err == nil {
		t.Fatal("expected error for missing hf_path")
	}
}

func TestValidation_MissingCategory(t *testing.T) {
	toml := `
[tokenizer]
path = "/tmp/tok"
[[dataset]]
name = "x"
hf_path = "org/repo"
`
	_, err := Load(writeTempTOML(t, toml))
	if err == nil {
		t.Fatal("expected error for missing category")
	}
}

func TestResolveScript(t *testing.T) {
	cfg := &Config{}
	got := cfg.ResolveScript("/data/pipeline.toml", "scripts/foo.lua")
	want := "/data/scripts/foo.lua"
	if got != want {
		t.Errorf("ResolveScript = %q, want %q", got, want)
	}
	// Absolute path passes through
	abs := cfg.ResolveScript("/data/pipeline.toml", "/abs/foo.lua")
	if abs != "/abs/foo.lua" {
		t.Errorf("ResolveScript(abs) = %q", abs)
	}
}

func TestScriptDir(t *testing.T) {
	cfg := &Config{}
	got := cfg.ScriptDir("/data/pipeline.toml")
	if got != "/data/scripts" {
		t.Errorf("ScriptDir = %q", got)
	}
}

func TestLoadMissingFile(t *testing.T) {
	_, err := Load("/nonexistent/pipeline.toml")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadInvalidTOML(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.toml")
	os.WriteFile(p, []byte(`[broken`), 0644)
	_, err := Load(p)
	if err == nil {
		t.Fatal("expected error for invalid TOML")
	}
}
