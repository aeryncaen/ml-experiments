// Package tokenize handles BPE tokenization and shard writing.
//
// The fast encoder is ported from experiments/yamit/tokenizer/main.go.
// It bypasses the slow tokenizer path by directly running:
// added-token split → regex split → byte-level map → BPE encode.
package tokenize

import (
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/model/bpe"
	"github.com/sugarme/tokenizer/normalizer"
	"github.com/sugarme/tokenizer/pretokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	"golang.org/x/text/unicode/norm"
)

const fastPathVerifyDocs = 8

type addedTokenEntry struct {
	content string
	id      int
}

// FastEncoder runs the regex-split + byte-level-map + BPE pipeline directly,
// bypassing the general tokenizer path for ~3-5x speedup.
type FastEncoder struct {
	model        *bpe.BPE
	splitPattern normalizer.Pattern
	byteLevel    *pretokenizer.ByteLevel
	byteChar     [256]string
	normalizeNFC bool
	verifyBudget int
	addedTokens  []addedTokenEntry // sorted longest-first

	// Fallback slow tokenizer.
	tk *tokenizer.Tokenizer
}

// LoadEncoder loads a tokenizer from file and builds a FastEncoder.
// If the fast path can't be used, falls back to slow encoding.
func LoadEncoder(tokenizerPath string, bpeCacheCapacity int) (*FastEncoder, error) {
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	// Optionally override BPE cache.
	if bpeCacheCapacity >= 0 {
		if m, ok := tk.GetModel().(*bpe.BPE); ok {
			if bpeCacheCapacity == 0 {
				m.Cache = nil
			} else {
				m.Cache = bpe.NewCache(bpeCacheCapacity)
			}
		}
	}

	fe, reason := buildFastEncoder(tk)
	if fe != nil {
		log.Printf("[encoder] fast path enabled (verify_docs=%d)", fe.verifyBudget)
		return fe, nil
	}

	log.Printf("[encoder] fast path disabled: %s; using slow path", reason)
	return &FastEncoder{tk: tk}, nil
}

func buildFastEncoder(tk *tokenizer.Tokenizer) (*FastEncoder, string) {
	model, ok := tk.GetModel().(*bpe.BPE)
	if !ok {
		return nil, "model is not BPE"
	}

	seq, ok := tk.GetPreTokenizer().(*pretokenizer.Sequence)
	if !ok {
		return nil, "pretokenizer is not Sequence"
	}

	pretoks := seq.PreTokenizers()
	if len(pretoks) != 2 {
		return nil, fmt.Sprintf("pretokenizer sequence len=%d (need 2)", len(pretoks))
	}

	split, ok := pretoks[0].(*pretokenizer.Split)
	if !ok {
		return nil, "first pretokenizer is not Split"
	}
	if split.Invert {
		return nil, "split invert=true unsupported"
	}
	if split.Behavior != normalizer.IsolatedBehavior {
		return nil, "split behavior is not Isolated"
	}

	bl, ok := pretoks[1].(*pretokenizer.ByteLevel)
	if !ok {
		return nil, "second pretokenizer is not ByteLevel"
	}
	if bl.UseRegex {
		return nil, "bytelevel use_regex=true unsupported"
	}

	normer := tk.GetNormalizer()
	normalizeNFC := false
	if normer != nil {
		if _, ok := normer.(*normalizer.NFC); ok {
			normalizeNFC = true
		} else {
			return nil, fmt.Sprintf("normalizer type %T unsupported", normer)
		}
	}

	// Collect added tokens for splitting before BPE.
	addedVocab := tk.GetAddedVocab()
	entries := make([]addedTokenEntry, 0, len(addedVocab))
	for content, id := range addedVocab {
		entries = append(entries, addedTokenEntry{content, id})
	}
	// Sort longest-first for greedy matching.
	sort.Slice(entries, func(i, j int) bool {
		if len(entries[i].content) != len(entries[j].content) {
			return len(entries[i].content) > len(entries[j].content)
		}
		return entries[i].content < entries[j].content
	})

	fe := &FastEncoder{
		model:        model,
		splitPattern: split.Pattern,
		byteLevel:    bl,
		normalizeNFC: normalizeNFC,
		verifyBudget: fastPathVerifyDocs,
		addedTokens:  entries,
		tk:           tk,
	}
	for i := 0; i < 256; i++ {
		fe.byteChar[i] = pretokenizer.BytesChar[byte(i)]
	}

	return fe, ""
}

func (f *FastEncoder) byteLevelMap(s string) string {
	if len(s) == 0 {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) * 2)
	for i := 0; i < len(s); i++ {
		b.WriteString(f.byteChar[s[i]])
	}
	return b.String()
}

// encodeBPESegment runs the regex-split + byte-level-map + BPE pipeline on
// a segment of text that is known to contain no added tokens.
func (f *FastEncoder) encodeBPESegment(segment string, dst []int) []int {
	matches := f.splitPattern.FindMatches(segment)
	for _, m := range matches {
		start := m.Offsets[0]
		end := m.Offsets[1]
		if start < 0 || end > len(segment) || start >= end {
			continue
		}

		piece := segment[start:end]
		if f.byteLevel.AddPrefixSpace && !strings.HasPrefix(piece, " ") {
			piece = " " + piece
		}

		mapped := f.byteLevelMap(piece)
		toks := f.model.TokenizeWithCache(mapped)
		for _, tok := range toks {
			dst = append(dst, tok.Id)
		}
	}
	return dst
}

// Encode tokenizes text and returns token IDs.
func (f *FastEncoder) Encode(text string) ([]int, error) {
	ids, err := f.encodeIntoIDs(text, make([]int, 0, 256))
	return ids, err
}

// EncodeInto tokenizes text into the provided slice (reset to len 0).
func (f *FastEncoder) EncodeInto(text string, dst []int) ([]int, error) {
	return f.encodeIntoIDs(text, dst)
}

func (f *FastEncoder) encodeIntoIDs(text string, dst []int) ([]int, error) {
	normText := text
	if f.normalizeNFC {
		normText = norm.NFC.String(text)
	}

	dst = dst[:0]

	// If fast path is available (model != nil), use it.
	if f.model != nil {
		dst = f.encodeFast(normText, dst)

		// Verify against slow path for first N documents.
		if f.verifyBudget > 0 {
			slowIDs, err := f.encodeSlow(normText)
			if err != nil {
				log.Printf("WARNING: fast-path verify slow-encode failed: %v", err)
			} else if !equalIntSlices(dst, slowIDs) {
				log.Printf("WARNING: fast-path mismatch; disabling fast path")
				f.model = nil
				return slowIDs, nil
			} else {
				f.verifyBudget--
			}
		}
		return dst, nil
	}

	// Slow path fallback.
	return f.encodeSlow(normText)
}

func (f *FastEncoder) encodeFast(text string, dst []int) []int {
	if len(f.addedTokens) == 0 {
		return f.encodeBPESegment(text, dst)
	}

	// Split text around added tokens, greedy longest-match scan.
	remaining := text
	for len(remaining) > 0 {
		bestIdx := -1
		bestPos := -1
		for i, at := range f.addedTokens {
			pos := strings.Index(remaining, at.content)
			if pos != -1 && (bestPos == -1 || pos < bestPos) {
				bestPos = pos
				bestIdx = i
				break // sorted longest-first, first hit at this pos is best
			}
		}
		if bestIdx == -1 {
			dst = f.encodeBPESegment(remaining, dst)
			break
		}

		if bestPos > 0 {
			dst = f.encodeBPESegment(remaining[:bestPos], dst)
		}
		dst = append(dst, f.addedTokens[bestIdx].id)
		remaining = remaining[bestPos+len(f.addedTokens[bestIdx].content):]
	}

	return dst
}

func (f *FastEncoder) encodeSlow(text string) ([]int, error) {
	enc, err := f.tk.EncodeSingle(text)
	if err != nil {
		return nil, err
	}
	return enc.Ids, nil
}

// VocabSize returns the tokenizer vocabulary size.
func (f *FastEncoder) VocabSize() int {
	return f.tk.GetVocabSize(true)
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
