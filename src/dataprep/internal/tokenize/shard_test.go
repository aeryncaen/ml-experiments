package tokenize

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestPTShardWriter(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "shard_0000")

	w, err := NewShardWriter(prefix)
	if err != nil {
		t.Fatal(err)
	}

	// Write two documents.
	doc1 := []int{100, 200, 300}
	doc2 := []int{400, 500}
	if err := w.WriteDocument(doc1); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteDocument(doc2); err != nil {
		t.Fatal(err)
	}

	if w.DocCount != 2 {
		t.Errorf("DocCount = %d", w.DocCount)
	}
	if w.TokenCount != 5 {
		t.Errorf("TokenCount = %d", w.TokenCount)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify .bin file.
	binData, err := os.ReadFile(prefix + ".bin")
	if err != nil {
		t.Fatal(err)
	}
	if len(binData) != 5*4 {
		t.Fatalf("bin size = %d, want %d", len(binData), 5*4)
	}
	// Read back token IDs.
	ids := make([]uint32, 5)
	for i := range ids {
		ids[i] = binary.LittleEndian.Uint32(binData[i*4:])
	}
	want := []uint32{100, 200, 300, 400, 500}
	for i, id := range ids {
		if id != want[i] {
			t.Errorf("token[%d] = %d, want %d", i, id, want[i])
		}
	}

	// Verify .idx file: should have 3 uint64 entries (offsets: 0, 12, 20).
	idxData, err := os.ReadFile(prefix + ".idx")
	if err != nil {
		t.Fatal(err)
	}
	if len(idxData) != 3*8 {
		t.Fatalf("idx size = %d, want %d", len(idxData), 3*8)
	}
	offsets := make([]uint64, 3)
	for i := range offsets {
		offsets[i] = binary.LittleEndian.Uint64(idxData[i*8:])
	}
	wantOffsets := []uint64{0, 12, 20}
	for i, off := range offsets {
		if off != wantOffsets[i] {
			t.Errorf("offset[%d] = %d, want %d", i, off, wantOffsets[i])
		}
	}

	// No .mask file for PT shards.
	if _, err := os.Stat(prefix + ".mask"); !os.IsNotExist(err) {
		t.Error("PT shard should not have .mask file")
	}
}

func TestSFTShardWriter(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "sft_0000")

	w, err := NewSFTShardWriter(prefix)
	if err != nil {
		t.Fatal(err)
	}

	tokens := []int{10, 20, 30, 40, 50}
	mask := []byte{0, 0, 1, 1, 1}
	if err := w.WriteSFTDocument(tokens, mask); err != nil {
		t.Fatal(err)
	}

	if w.DocCount != 1 {
		t.Errorf("DocCount = %d", w.DocCount)
	}
	if w.TokenCount != 5 {
		t.Errorf("TokenCount = %d", w.TokenCount)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify .bin
	binData, err := os.ReadFile(prefix + ".bin")
	if err != nil {
		t.Fatal(err)
	}
	if len(binData) != 5*4 {
		t.Fatalf("bin size = %d", len(binData))
	}

	// Verify .mask
	maskData, err := os.ReadFile(prefix + ".mask")
	if err != nil {
		t.Fatal(err)
	}
	if len(maskData) != 5 {
		t.Fatalf("mask size = %d", len(maskData))
	}
	wantMask := []byte{0, 0, 1, 1, 1}
	for i, b := range maskData {
		if b != wantMask[i] {
			t.Errorf("mask[%d] = %d, want %d", i, b, wantMask[i])
		}
	}

	// No .idx file for SFT shards.
	if _, err := os.Stat(prefix + ".idx"); !os.IsNotExist(err) {
		t.Error("SFT shard should not have .idx file")
	}
}

func TestSFTMismatchedLengths(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "sft_bad")

	w, err := NewSFTShardWriter(prefix)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	err = w.WriteSFTDocument([]int{1, 2, 3}, []byte{0, 1})
	if err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
}

func TestPTWriteSFTFails(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "pt_no_sft")

	w, err := NewShardWriter(prefix)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()

	err = w.WriteSFTDocument([]int{1}, []byte{1})
	if err == nil {
		t.Fatal("expected error writing SFT to PT shard")
	}
}

func TestEmptyDocument(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "empty")

	w, err := NewShardWriter(prefix)
	if err != nil {
		t.Fatal(err)
	}

	if err := w.WriteDocument([]int{}); err != nil {
		t.Fatal(err)
	}
	if w.DocCount != 1 {
		t.Errorf("DocCount = %d", w.DocCount)
	}
	if w.TokenCount != 0 {
		t.Errorf("TokenCount = %d", w.TokenCount)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestMultipleDocuments(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "multi")

	w, err := NewShardWriter(prefix)
	if err != nil {
		t.Fatal(err)
	}

	// Write 100 documents of varying sizes.
	totalTokens := 0
	for i := 0; i < 100; i++ {
		size := (i % 10) + 1
		doc := make([]int, size)
		for j := range doc {
			doc[j] = i*1000 + j
		}
		if err := w.WriteDocument(doc); err != nil {
			t.Fatal(err)
		}
		totalTokens += size
	}

	if w.DocCount != 100 {
		t.Errorf("DocCount = %d", w.DocCount)
	}
	if int(w.TokenCount) != totalTokens {
		t.Errorf("TokenCount = %d, want %d", w.TokenCount, totalTokens)
	}

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Verify bin size.
	binData, err := os.ReadFile(prefix + ".bin")
	if err != nil {
		t.Fatal(err)
	}
	if len(binData) != totalTokens*4 {
		t.Errorf("bin size = %d, want %d", len(binData), totalTokens*4)
	}

	// Verify idx has 101 entries (initial 0 + 100 doc end offsets).
	idxData, err := os.ReadFile(prefix + ".idx")
	if err != nil {
		t.Fatal(err)
	}
	if len(idxData) != 101*8 {
		t.Errorf("idx size = %d, want %d", len(idxData), 101*8)
	}
}

func TestRemove(t *testing.T) {
	dir := t.TempDir()
	prefix := filepath.Join(dir, "removable")

	w, err := NewShardWriter(prefix)
	if err != nil {
		t.Fatal(err)
	}
	w.WriteDocument([]int{1, 2, 3})
	w.Close()

	// Files should exist.
	if _, err := os.Stat(prefix + ".bin"); err != nil {
		t.Error("bin should exist before Remove")
	}
	if _, err := os.Stat(prefix + ".idx"); err != nil {
		t.Error("idx should exist before Remove")
	}

	w.Remove(prefix)

	// Files should be gone.
	if _, err := os.Stat(prefix + ".bin"); !os.IsNotExist(err) {
		t.Error("bin should not exist after Remove")
	}
	if _, err := os.Stat(prefix + ".idx"); !os.IsNotExist(err) {
		t.Error("idx should not exist after Remove")
	}
}
