// Package main implements mass-parallel tokenization of JSONL text shards
// into uint32 binary token shards using the Qwen3 tokenizer.
//
// It reads JSONL files ({"text": "..."} per line), tokenizes each document,
// applies long-token surgery (remapping tokens > 16 bytes to byte-level
// fallback sequences), and writes binary output shards.
//
// Output format:
//   - .bin files: flat arrays of little-endian uint32 token IDs
//   - .idx files: little-endian uint64 byte offsets marking document boundaries
//     in the corresponding .bin file (N+1 entries for N documents)
//
// Usage:
//
//	go run . \
//	  --tokenizer ./tokenizer.json \
//	  --input ./raw_data \
//	  --output ./tokenized \
//	  --workers 16 \
//	  --val-fraction 0.005
package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

// ── Surgery table ────────────────────────────────────────────────────────

const maxBytesPerToken = 16

// surgeryTable maps token IDs whose decoded byte representation exceeds
// maxBytesPerToken to their byte-level fallback token sequences.
type surgeryTable struct {
	remap map[int][]int // long_token_id → []byte_token_id
}

func loadSurgeryTable(path string) (*surgeryTable, error) {
	if path == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read surgery map: %w", err)
	}

	var raw map[string][]int
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("parse surgery map json: %w", err)
	}

	remap := make(map[int][]int, len(raw))
	for k, v := range raw {
		oldID, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("invalid surgery key %q: %w", k, err)
		}
		remap[oldID] = v
	}
	return &surgeryTable{remap: remap}, nil
}

// buildSurgeryTable examines every token in the vocabulary. Tokens whose
// UTF-8 byte length exceeds maxBytesPerToken are decomposed into a sequence
// of byte-level tokens (<0x00> through <0xFF>).
func buildSurgeryTable(tk *tokenizer.Tokenizer) (*surgeryTable, error) {
	vocab := tk.GetVocab(true) // map[string]int — token string → ID
	vocabSize := tk.GetVocabSize(true)

	// Build reverse map: ID → token string.
	idToToken := make(map[int]string, len(vocab))
	for tok, id := range vocab {
		idToToken[id] = tok
	}

	// Find byte-level token IDs: <0x00> through <0xFF>.
	byteTokenIDs := make(map[byte]int, 256)
	for b := 0; b < 256; b++ {
		hexToken := fmt.Sprintf("<0x%02X>", b)
		id, ok := vocab[hexToken]
		if !ok {
			// Try lowercase variant.
			hexToken = fmt.Sprintf("<0x%02x>", b)
			id, ok = vocab[hexToken]
		}
		if ok {
			byteTokenIDs[byte(b)] = id
		}
	}

	if len(byteTokenIDs) == 0 {
		// Tokenizer might use a different byte fallback scheme.
		// For tiktoken-style tokenizers, single-byte tokens ARE the byte
		// fallback. We need to identify them differently.
		log.Printf("WARNING: No <0xHH> byte tokens found in vocabulary. " +
			"Attempting to identify single-byte tokens from vocab.")
		for id := 0; id < vocabSize; id++ {
			tok, ok := idToToken[id]
			if !ok {
				continue
			}
			b := []byte(tok)
			if len(b) == 1 {
				byteTokenIDs[b[0]] = id
			}
		}
		log.Printf("Found %d single-byte tokens", len(byteTokenIDs))
	}

	if len(byteTokenIDs) < 256 {
		return nil, fmt.Errorf(
			"only found %d/256 byte-level tokens — cannot perform surgery",
			len(byteTokenIDs),
		)
	}

	// Find tokens that need surgery.
	remap := make(map[int][]int)
	for id := 0; id < vocabSize; id++ {
		tok, ok := idToToken[id]
		if !ok {
			continue
		}
		tokBytes := []byte(tok)
		if len(tokBytes) > maxBytesPerToken {
			// Decompose into byte-level token sequence.
			byteSeq := make([]int, len(tokBytes))
			for i, b := range tokBytes {
				byteSeq[i] = byteTokenIDs[b]
			}
			remap[id] = byteSeq
		}
	}

	log.Printf("Surgery table: %d tokens > %d bytes remapped (vocab size: %d)",
		len(remap), maxBytesPerToken, vocabSize)

	return &surgeryTable{remap: remap}, nil
}

// apply takes a token ID sequence and expands any surgically-remapped tokens
// into their byte-level fallback sequences.
func (s *surgeryTable) apply(ids []int) []int {
	if len(s.remap) == 0 {
		return ids
	}
	// Pre-check: if no token needs surgery, return as-is (fast path).
	needsSurgery := false
	for _, id := range ids {
		if _, ok := s.remap[id]; ok {
			needsSurgery = true
			break
		}
	}
	if !needsSurgery {
		return ids
	}

	out := make([]int, 0, len(ids)+len(ids)/4) // slight overalloc
	for _, id := range ids {
		if expanded, ok := s.remap[id]; ok {
			out = append(out, expanded...)
		} else {
			out = append(out, id)
		}
	}
	return out
}

// ── Shard writer ─────────────────────────────────────────────────────────

// shardWriter writes tokenized documents to .bin/.idx file pairs.
type shardWriter struct {
	binFile    *os.File
	idxFile    *os.File
	binBuf     *bufio.Writer
	idxBuf     *bufio.Writer
	binOffset  uint64
	docCount   uint64
	tokenCount uint64
}

func newShardWriter(path string) (*shardWriter, error) {
	binPath := path + ".bin"
	idxPath := path + ".idx"

	binFile, err := os.Create(binPath)
	if err != nil {
		return nil, fmt.Errorf("create %s: %w", binPath, err)
	}
	idxFile, err := os.Create(idxPath)
	if err != nil {
		binFile.Close()
		return nil, fmt.Errorf("create %s: %w", idxPath, err)
	}

	w := &shardWriter{
		binFile: binFile,
		idxFile: idxFile,
		binBuf:  bufio.NewWriterSize(binFile, 1<<20), // 1MB buffer
		idxBuf:  bufio.NewWriterSize(idxFile, 1<<16),
	}

	// Write initial offset (0) for first document.
	if err := binary.Write(w.idxBuf, binary.LittleEndian, uint64(0)); err != nil {
		w.Close()
		return nil, err
	}

	return w, nil
}

// writeDocument appends a tokenized document's token IDs to the shard.
func (w *shardWriter) writeDocument(tokenIDs []int) error {
	for _, id := range tokenIDs {
		if err := binary.Write(w.binBuf, binary.LittleEndian, uint32(id)); err != nil {
			return err
		}
	}
	w.binOffset += uint64(len(tokenIDs)) * 4
	w.tokenCount += uint64(len(tokenIDs))
	w.docCount++

	// Write end offset for this document.
	return binary.Write(w.idxBuf, binary.LittleEndian, w.binOffset)
}

func (w *shardWriter) Close() error {
	var errs []error
	if err := w.binBuf.Flush(); err != nil {
		errs = append(errs, err)
	}
	if err := w.idxBuf.Flush(); err != nil {
		errs = append(errs, err)
	}
	if err := w.binFile.Close(); err != nil {
		errs = append(errs, err)
	}
	if err := w.idxFile.Close(); err != nil {
		errs = append(errs, err)
	}
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// ── JSONL reading ────────────────────────────────────────────────────────

type jsonlRecord struct {
	Text string `json:"text"`
}

func readJSONLFile(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var texts []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0), 10*1024*1024) // 10MB max line
	for scanner.Scan() {
		var rec jsonlRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			continue // skip malformed lines
		}
		if len(strings.TrimSpace(rec.Text)) > 0 {
			texts = append(texts, rec.Text)
		}
	}
	return texts, scanner.Err()
}

// ── Worker pool ──────────────────────────────────────────────────────────

type tokenizeResult struct {
	shardName  string
	tokenIDs   [][]int // one per document
	inputPath  string
	tokenCount int
	docCount   int
	err        error
}

func loadIDRemap(path string) (map[int]int, error) {
	if path == "" {
		return nil, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read id remap: %w", err)
	}

	var raw map[string]int
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, fmt.Errorf("parse id remap json: %w", err)
	}

	out := make(map[int]int, len(raw))
	for k, v := range raw {
		oldID, err := strconv.Atoi(k)
		if err != nil {
			return nil, fmt.Errorf("invalid id remap key %q: %w", k, err)
		}
		out[oldID] = v
	}
	return out, nil
}

func applyIDRemap(ids []int, remap map[int]int) ([]int, error) {
	if remap == nil {
		return ids, nil
	}
	out := make([]int, len(ids))
	for i, id := range ids {
		newID, ok := remap[id]
		if !ok {
			return nil, fmt.Errorf("missing remap for token id %d", id)
		}
		out[i] = newID
	}
	return out, nil
}

func tokenizeWorker(
	tk *tokenizer.Tokenizer,
	surgery *surgeryTable,
	idRemap map[int]int,
	jobs <-chan string,
	results chan<- tokenizeResult,
) {
	for path := range jobs {
		texts, err := readJSONLFile(path)
		if err != nil {
			results <- tokenizeResult{inputPath: path, err: err}
			continue
		}

		allIDs := make([][]int, 0, len(texts))
		totalTokens := 0
		var fileErr error

		for _, text := range texts {
			enc, err := tk.EncodeSingle(text)
			if err != nil {
				log.Printf("WARNING: failed to encode document in %s: %v", path, err)
				continue
			}
			ids := enc.Ids
			ids = surgery.apply(ids)
			ids, err = applyIDRemap(ids, idRemap)
			if err != nil {
				fileErr = err
				break
			}
			allIDs = append(allIDs, ids)
			totalTokens += len(ids)
		}

		if fileErr != nil {
			results <- tokenizeResult{inputPath: path, err: fileErr}
			continue
		}

		base := filepath.Base(path)
		shardName := strings.TrimSuffix(base, ".jsonl")

		results <- tokenizeResult{
			shardName:  shardName,
			tokenIDs:   allIDs,
			inputPath:  path,
			tokenCount: totalTokens,
			docCount:   len(allIDs),
		}
	}
}

// ── Main ─────────────────────────────────────────────────────────────────

func main() {
	tokenizerPath := flag.String("tokenizer", "", "Path to tokenizer.json")
	surgeryMapPath := flag.String("surgery-map", "", "Optional path to surgery_map.json")
	idRemapPath := flag.String("id-remap", "", "Optional path to id_remap.json (old tokenizer IDs -> model IDs)")
	inputDir := flag.String("input", "", "Input directory with JSONL shards (searched recursively)")
	outputDir := flag.String("output", "", "Output directory for .bin/.idx shards")
	workers := flag.Int("workers", 0, "Number of worker goroutines (default: NumCPU)")
	valFraction := flag.Float64("val-fraction", 0.005, "Fraction of shards to reserve for validation")
	flag.Parse()

	if *tokenizerPath == "" || *inputDir == "" || *outputDir == "" {
		flag.Usage()
		os.Exit(1)
	}
	if *workers <= 0 {
		*workers = runtime.NumCPU()
	}

	startTime := time.Now()

	// Load tokenizer.
	log.Printf("Loading tokenizer from %s", *tokenizerPath)
	tk, err := pretrained.FromFile(*tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	log.Printf("Vocabulary size: %d", tk.GetVocabSize(true))

	// Surgery table: prefer artifact file when provided.
	surgery, err := loadSurgeryTable(*surgeryMapPath)
	if err != nil {
		log.Fatalf("Failed to load surgery map: %v", err)
	}
	if surgery == nil {
		surgery, err = buildSurgeryTable(tk)
		if err != nil {
			log.Fatalf("Failed to build surgery table: %v", err)
		}
		log.Printf("Built surgery table from tokenizer vocab: %d entries", len(surgery.remap))
	} else {
		log.Printf("Loaded surgery map: %d entries", len(surgery.remap))
	}

	// Optional ID remap for pruned composite vocab.
	idRemap, err := loadIDRemap(*idRemapPath)
	if err != nil {
		log.Fatalf("Failed to load id remap: %v", err)
	}
	if idRemap != nil {
		log.Printf("Loaded ID remap: %d entries", len(idRemap))
	}

	// Find all JSONL files.
	var jsonlFiles []string
	err = filepath.Walk(*inputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(path, ".jsonl") {
			jsonlFiles = append(jsonlFiles, path)
		}
		return nil
	})
	if err != nil {
		log.Fatalf("Failed to walk input directory: %v", err)
	}
	sort.Strings(jsonlFiles)
	log.Printf("Found %d JSONL files", len(jsonlFiles))

	if len(jsonlFiles) == 0 {
		log.Fatal("No .jsonl files found")
	}

	// Create output directories.
	trainDir := filepath.Join(*outputDir, "train")
	valDir := filepath.Join(*outputDir, "val")
	os.MkdirAll(trainDir, 0o755)
	os.MkdirAll(valDir, 0o755)

	// Determine train/val split at the shard level.
	// Shuffle deterministically, then take the first N shards for val.
	rng := rand.New(rand.NewSource(42))
	indices := make([]int, len(jsonlFiles))
	for i := range indices {
		indices[i] = i
	}
	rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

	valCount := int(float64(len(jsonlFiles)) * *valFraction)
	if valCount < 1 && len(jsonlFiles) > 1 {
		valCount = 1
	}
	valSet := make(map[int]bool, valCount)
	for i := 0; i < valCount; i++ {
		valSet[indices[i]] = true
	}

	// Launch worker pool.
	jobs := make(chan string, *workers*2)
	results := make(chan tokenizeResult, *workers*2)

	var wg sync.WaitGroup
	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tokenizeWorker(tk, surgery, idRemap, jobs, results)
		}()
	}

	// Feed jobs.
	go func() {
		for _, path := range jsonlFiles {
			jobs <- path
		}
		close(jobs)
	}()

	// Collect results in a separate goroutine.
	go func() {
		wg.Wait()
		close(results)
	}()

	// Process results and write shards.
	var totalTokens atomic.Uint64
	var totalDocs atomic.Uint64
	var trainTokens, valTokens uint64
	var errorCount int

	// Map original file index to its result for ordered writing.
	fileToIdx := make(map[string]int, len(jsonlFiles))
	for i, f := range jsonlFiles {
		fileToIdx[f] = i
	}

	for result := range results {
		if result.err != nil {
			log.Printf("ERROR processing %s: %v", result.inputPath, result.err)
			errorCount++
			continue
		}

		idx := fileToIdx[result.inputPath]
		isVal := valSet[idx]

		var outDir string
		if isVal {
			outDir = valDir
		} else {
			outDir = trainDir
		}

		// Derive output shard name from the dataset subdirectory + shard name.
		relPath, _ := filepath.Rel(*inputDir, result.inputPath)
		relDir := filepath.Dir(relPath)
		shardOutDir := filepath.Join(outDir, relDir)
		os.MkdirAll(shardOutDir, 0o755)
		shardPath := filepath.Join(shardOutDir, result.shardName)

		writer, err := newShardWriter(shardPath)
		if err != nil {
			log.Printf("ERROR creating shard writer for %s: %v", shardPath, err)
			errorCount++
			continue
		}

		for _, docIDs := range result.tokenIDs {
			if err := writer.writeDocument(docIDs); err != nil {
				log.Printf("ERROR writing document to %s: %v", shardPath, err)
				break
			}
		}
		writer.Close()

		totalTokens.Add(uint64(result.tokenCount))
		totalDocs.Add(uint64(result.docCount))
		if isVal {
			valTokens += uint64(result.tokenCount)
		} else {
			trainTokens += uint64(result.tokenCount)
		}

		split := "train"
		if isVal {
			split = "val  "
		}
		log.Printf("[%s] %s — %d docs, %d tokens",
			split, relPath, result.docCount, result.tokenCount)
	}

	elapsed := time.Since(startTime)
	tTotal := totalTokens.Load()
	dTotal := totalDocs.Load()

	log.Printf("")
	log.Printf("═══════════════════════════════════════════════════")
	log.Printf("Tokenization complete in %v", elapsed.Round(time.Second))
	log.Printf("  Total:  %d documents, %d tokens", dTotal, tTotal)
	log.Printf("  Train:  %d tokens", trainTokens)
	log.Printf("  Val:    %d tokens", valTokens)
	log.Printf("  Errors: %d", errorCount)
	if elapsed.Seconds() > 0 {
		log.Printf("  Speed:  %.0f tokens/sec", float64(tTotal)/elapsed.Seconds())
	}
	log.Printf("═══════════════════════════════════════════════════")

	// Write tokenization metadata.
	metaPath := filepath.Join(*outputDir, "tokenize_meta.json")
	meta := map[string]interface{}{
		"tokenizer":     *tokenizerPath,
		"vocab_size":    tk.GetVocabSize(true),
		"surgery_count": len(surgery.remap),
		"max_bytes":     maxBytesPerToken,
		"total_tokens":  tTotal,
		"total_docs":    dTotal,
		"train_tokens":  trainTokens,
		"val_tokens":    valTokens,
		"val_fraction":  *valFraction,
		"workers":       *workers,
		"elapsed_secs":  elapsed.Seconds(),
		"errors":        errorCount,
	}
	metaJSON, _ := json.MarshalIndent(meta, "", "  ")
	os.WriteFile(metaPath, metaJSON, 0o644)
	log.Printf("Metadata written to %s", metaPath)
}

// Ensure io import is used (for potential future streaming).
var _ = io.EOF
