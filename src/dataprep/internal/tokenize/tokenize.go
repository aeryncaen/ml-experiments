package tokenize

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/config"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/process"
	"github.com/bzoidberg/heuristic-secrets/src/dataprep/internal/state"
)

// ChatML special tokens for SFT formatting.
const (
	chatMLSystem    = "<|im_start|>system\n"
	chatMLUser      = "<|im_start|>user\n"
	chatMLAssistant = "<|im_start|>assistant\n"
	chatMLEnd       = "<|im_end|>\n"
)

// Tokenizer orchestrates tokenization of processed JSONL into .bin/.idx/.mask shards.
type Tokenizer struct {
	cfg        *config.Config
	db         *state.DB
	configPath string
}

// New creates a new Tokenizer.
func New(cfg *config.Config, db *state.DB, configPath string) *Tokenizer {
	return &Tokenizer{cfg: cfg, db: db, configPath: configPath}
}

// Run tokenizes processed JSONL shards for the given datasets.
func (t *Tokenizer) Run(datasets []config.DatasetConfig) error {
	for _, ds := range datasets {
		if err := t.tokenizeDataset(ds); err != nil {
			return fmt.Errorf("tokenize %s: %w", ds.Name, err)
		}
	}
	return nil
}

type tokenizeJob struct {
	shard    state.Shard
	ds       config.DatasetConfig
	trainDir string
	valDir   string
	sftDir   string
}

type tokenizeResult struct {
	shardIdx    int
	dataset     string
	trainTokens int64
	trainDocs   int64
	valTokens   int64
	valDocs     int64
	sftTokens   int64
	sftDocs     int64
	elapsed     time.Duration
	err         error
}

func (t *Tokenizer) tokenizeDataset(ds config.DatasetConfig) error {
	processedDir := filepath.Join(t.cfg.Output.ProcessedDir, ds.Name)

	// Find processed JSONL shards.
	entries, err := os.ReadDir(processedDir)
	if err != nil {
		return fmt.Errorf("read processed dir %s: %w (run process first?)", processedDir, err)
	}

	var processedFiles []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
			processedFiles = append(processedFiles, filepath.Join(processedDir, e.Name()))
		}
	}
	if len(processedFiles) == 0 {
		return fmt.Errorf("no processed JSONL shards found in %s", processedDir)
	}

	// Output directories.
	trainDir := filepath.Join(t.cfg.Output.TrainDir, ds.Name)
	valDir := filepath.Join(t.cfg.Output.ValDir, ds.Name)
	sftDir := filepath.Join(t.cfg.Output.SFTDir, ds.Name)

	os.MkdirAll(trainDir, 0o755)
	os.MkdirAll(valDir, 0o755)
	if ds.SFTCapable {
		os.MkdirAll(sftDir, 0o755)
	}

	// Register shards.
	for i, pFile := range processedFiles {
		t.db.EnsureShard(ds.Name, i, state.StageTokenize, pFile)
	}

	pending, err := t.db.PendingShards(ds.Name, state.StageTokenize)
	if err != nil {
		return err
	}

	if len(pending) == 0 {
		log.Printf("[tokenize] %s: all %d shards already done, skipping", ds.Name, len(processedFiles))
		return nil
	}

	log.Printf("[tokenize] %s: %d/%d shards to tokenize (%d workers, sft=%v)",
		ds.Name, len(pending), len(processedFiles), t.cfg.Defaults.Workers, ds.SFTCapable)

	// Worker pool.
	numWorkers := t.cfg.Defaults.Workers
	if numWorkers > len(pending) {
		numWorkers = len(pending)
	}
	if numWorkers < 1 {
		numWorkers = 1
	}

	jobs := make(chan tokenizeJob, numWorkers*2)
	results := make(chan tokenizeResult, numWorkers*2)

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			t.tokenizeWorker(jobs, results)
		}()
	}

	// Feed jobs.
	go func() {
		for _, sh := range pending {
			jobs <- tokenizeJob{
				shard:    sh,
				ds:       ds,
				trainDir: trainDir,
				valDir:   valDir,
				sftDir:   sftDir,
			}
		}
		close(jobs)
	}()

	// Collect results.
	go func() {
		wg.Wait()
		close(results)
	}()

	startTime := time.Now()
	var totalTokens atomic.Int64
	completed := 0

	for res := range results {
		completed++
		tokens := res.trainTokens + res.valTokens + res.sftTokens
		totalTokens.Add(tokens)

		if res.err != nil {
			log.Printf("[tokenize] %s shard %d: ERROR %v", res.dataset, res.shardIdx, res.err)
			t.db.MarkFailed(res.dataset, res.shardIdx, state.StageTokenize, res.err.Error())
			continue
		}

		outPath := filepath.Join(trainDir, fmt.Sprintf("shard_%05d", res.shardIdx))
		t.db.MarkDone(res.dataset, res.shardIdx, state.StageTokenize, outPath, res.trainTokens+res.valTokens, res.trainDocs+res.valDocs, 0)

		wallElapsed := time.Since(startTime).Seconds()
		aggTokSec := float64(0)
		if wallElapsed > 0 {
			aggTokSec = float64(totalTokens.Load()) / wallElapsed
		}

		log.Printf("[tokenize] %s shard %d: %dk tok (train=%d val=%d sft=%d), %.1fM tok/s agg, %d/%d",
			res.dataset, res.shardIdx, tokens/1000,
			res.trainTokens, res.valTokens, res.sftTokens,
			aggTokSec/1e6, completed, len(pending))
	}

	return nil
}

func (t *Tokenizer) tokenizeWorker(jobs <-chan tokenizeJob, results chan<- tokenizeResult) {
	var enc *FastEncoder
	var loadErr error

	for job := range jobs {
		if enc == nil && loadErr == nil {
			enc, loadErr = LoadEncoder(t.cfg.Tokenizer.Path, -1)
		}

		if loadErr != nil {
			results <- tokenizeResult{
				shardIdx: job.shard.ShardIdx,
				dataset:  job.ds.Name,
				err:      fmt.Errorf("load encoder: %w", loadErr),
			}
			continue
		}

		t0 := time.Now()
		res := t.tokenizeShard(job, enc)
		res.elapsed = time.Since(t0)
		results <- res
	}
}

func (t *Tokenizer) tokenizeShard(job tokenizeJob, enc *FastEncoder) tokenizeResult {
	res := tokenizeResult{
		shardIdx: job.shard.ShardIdx,
		dataset:  job.ds.Name,
	}

	t.db.MarkRunning(job.ds.Name, job.shard.ShardIdx, state.StageTokenize)

	// Open input.
	inFile, err := os.Open(job.shard.InputPath)
	if err != nil {
		res.err = err
		return res
	}
	defer inFile.Close()

	// Create shard writers.
	trainPrefix := filepath.Join(job.trainDir, fmt.Sprintf("shard_%05d", job.shard.ShardIdx))
	valPrefix := filepath.Join(job.valDir, fmt.Sprintf("shard_%05d", job.shard.ShardIdx))

	trainWriter, err := NewShardWriter(trainPrefix)
	if err != nil {
		res.err = err
		return res
	}
	defer trainWriter.Close()

	valWriter, err := NewShardWriter(valPrefix)
	if err != nil {
		res.err = err
		return res
	}
	defer valWriter.Close()

	var sftWriter *ShardWriter
	if job.ds.SFTCapable {
		sftPrefix := filepath.Join(job.sftDir, fmt.Sprintf("shard_%05d", job.shard.ShardIdx))
		sftWriter, err = NewSFTShardWriter(sftPrefix)
		if err != nil {
			res.err = err
			return res
		}
		defer sftWriter.Close()
	}

	// Process each record.
	scanner := bufio.NewScanner(inFile)
	scanner.Buffer(make([]byte, 0), 10*1024*1024)

	rng := rand.New(rand.NewSource(42 + int64(job.shard.ShardIdx)))
	tokenBuf := make([]int, 0, 256)
	eosTokenID := t.cfg.Tokenizer.EOSTokenID

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var rec process.ProcessedRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			continue
		}

		// PT tokenization (always, for all records).
		if rec.PTText != "" {
			tokenBuf, err = enc.EncodeInto(rec.PTText, tokenBuf)
			if err != nil {
				continue
			}
			if len(tokenBuf) == 0 {
				continue
			}

			// Append EOS.
			ids := make([]int, len(tokenBuf)+1)
			copy(ids, tokenBuf)
			ids[len(tokenBuf)] = eosTokenID

			// Route to train or val.
			if rec.Route == "val" || rng.Float64() < job.ds.ValFraction {
				if err := valWriter.WriteDocument(ids); err != nil {
					res.err = err
					return res
				}
				res.valTokens += int64(len(ids))
				res.valDocs++
			} else {
				if err := trainWriter.WriteDocument(ids); err != nil {
					res.err = err
					return res
				}
				res.trainTokens += int64(len(ids))
				res.trainDocs++
			}
		}

		// SFT tokenization (only for SFT-capable datasets with segments).
		if sftWriter != nil && len(rec.SFTSegments) > 0 {
			ids, mask, err := t.tokenizeChatML(enc, rec.SFTSegments, eosTokenID)
			if err != nil || len(ids) == 0 {
				continue
			}

			if err := sftWriter.WriteSFTDocument(ids, mask); err != nil {
				res.err = err
				return res
			}
			res.sftTokens += int64(len(ids))
			res.sftDocs++
		}
	}

	if err := scanner.Err(); err != nil {
		res.err = err
		return res
	}

	// Clean up empty shards.
	trainWriter.Close()
	valWriter.Close()
	if sftWriter != nil {
		sftWriter.Close()
	}

	if trainWriter.DocCount == 0 {
		os.Remove(trainPrefix + ".bin")
		os.Remove(trainPrefix + ".idx")
	}
	if valWriter.DocCount == 0 {
		os.Remove(valPrefix + ".bin")
		os.Remove(valPrefix + ".idx")
	}
	if sftWriter != nil && sftWriter.DocCount == 0 {
		os.Remove(filepath.Join(job.sftDir, fmt.Sprintf("shard_%05d.bin", job.shard.ShardIdx)))
		os.Remove(filepath.Join(job.sftDir, fmt.Sprintf("shard_%05d.mask", job.shard.ShardIdx)))
	}

	return res
}

// tokenizeChatML formats SFT segments as ChatML and produces token IDs + loss mask.
//
// Format:
//
//	<|im_start|>system\n{content}<|im_end|>\n
//	<|im_start|>user\n{content}<|im_end|>\n
//	<|im_start|>assistant\n{content}<|im_end|>\n
//
// mask[i] = 1 only for assistant completion tokens (where loss is active).
func (t *Tokenizer) tokenizeChatML(enc *FastEncoder, segments []process.SFTSegment, eosTokenID int) ([]int, []byte, error) {
	var allIDs []int
	var allMask []byte

	for _, seg := range segments {
		// Build the ChatML header.
		var header string
		switch seg.Role {
		case "system":
			header = chatMLSystem
		case "user":
			header = chatMLUser
		case "assistant":
			header = chatMLAssistant
		default:
			header = fmt.Sprintf("<|im_start|>%s\n", seg.Role)
		}

		// Tokenize header.
		headerIDs, err := enc.Encode(header)
		if err != nil {
			return nil, nil, err
		}

		// Tokenize content.
		contentIDs, err := enc.Encode(seg.Content)
		if err != nil {
			return nil, nil, err
		}

		// Tokenize end marker.
		endIDs, err := enc.Encode(chatMLEnd)
		if err != nil {
			return nil, nil, err
		}

		// Append header (always masked).
		allIDs = append(allIDs, headerIDs...)
		for range headerIDs {
			allMask = append(allMask, 0)
		}

		// Append content.
		allIDs = append(allIDs, contentIDs...)
		maskVal := byte(0)
		if seg.Loss {
			maskVal = 1
		}
		for range contentIDs {
			allMask = append(allMask, maskVal)
		}

		// Append end marker (same loss as content for assistant, masked otherwise).
		allIDs = append(allIDs, endIDs...)
		for range endIDs {
			allMask = append(allMask, maskVal)
		}
	}

	// Append EOS (masked from loss).
	allIDs = append(allIDs, eosTokenID)
	allMask = append(allMask, 0)

	return allIDs, allMask, nil
}
