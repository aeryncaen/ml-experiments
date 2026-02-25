package tokenize

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
)

// ShardWriter writes tokenized documents to .bin/.idx file pairs.
// Optionally writes a .mask sidecar for SFT loss masks.
type ShardWriter struct {
	binFile    *os.File
	idxFile    *os.File
	maskFile   *os.File // nil for PT-only shards
	binBuf     *bufio.Writer
	idxBuf     *bufio.Writer
	maskBuf    *bufio.Writer // nil for PT-only shards
	binScratch []byte
	idxScratch [8]byte
	binOffset  uint64
	DocCount   uint64
	TokenCount uint64
}

// NewShardWriter creates a new shard writer for PT output (.bin + .idx).
func NewShardWriter(pathPrefix string) (*ShardWriter, error) {
	return newShardWriter(pathPrefix, false)
}

// NewSFTShardWriter creates a new shard writer for SFT output (.bin + .mask).
func NewSFTShardWriter(pathPrefix string) (*ShardWriter, error) {
	return newShardWriter(pathPrefix, true)
}

func newShardWriter(pathPrefix string, withMask bool) (*ShardWriter, error) {
	binPath := pathPrefix + ".bin"
	binFile, err := os.Create(binPath)
	if err != nil {
		return nil, fmt.Errorf("create %s: %w", binPath, err)
	}

	w := &ShardWriter{
		binFile: binFile,
		binBuf:  bufio.NewWriterSize(binFile, 1<<20), // 1MB buffer
	}

	if withMask {
		// SFT: .bin + .mask (no .idx needed)
		maskPath := pathPrefix + ".mask"
		maskFile, err := os.Create(maskPath)
		if err != nil {
			binFile.Close()
			return nil, fmt.Errorf("create %s: %w", maskPath, err)
		}
		w.maskFile = maskFile
		w.maskBuf = bufio.NewWriterSize(maskFile, 1<<20)
	} else {
		// PT: .bin + .idx
		idxPath := pathPrefix + ".idx"
		idxFile, err := os.Create(idxPath)
		if err != nil {
			binFile.Close()
			return nil, fmt.Errorf("create %s: %w", idxPath, err)
		}
		w.idxFile = idxFile
		w.idxBuf = bufio.NewWriterSize(idxFile, 1<<16)

		// Write initial offset (0) for first document.
		if err := w.writeIndexOffset(0); err != nil {
			w.Close()
			return nil, err
		}
	}

	return w, nil
}

func (w *ShardWriter) writeIndexOffset(off uint64) error {
	binary.LittleEndian.PutUint64(w.idxScratch[:], off)
	_, err := w.idxBuf.Write(w.idxScratch[:])
	return err
}

// WriteDocument appends a tokenized document's token IDs to the PT shard.
func (w *ShardWriter) WriteDocument(tokenIDs []int) error {
	nBytes := len(tokenIDs) * 4
	if cap(w.binScratch) < nBytes {
		w.binScratch = make([]byte, nBytes)
	}
	buf := w.binScratch[:nBytes]
	for i, id := range tokenIDs {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(id))
	}
	if _, err := w.binBuf.Write(buf); err != nil {
		return err
	}
	w.binOffset += uint64(len(tokenIDs)) * 4
	w.TokenCount += uint64(len(tokenIDs))
	w.DocCount++

	// Write end offset for this document.
	if w.idxBuf != nil {
		return w.writeIndexOffset(w.binOffset)
	}
	return nil
}

// WriteSFTDocument appends tokenized SFT data to the shard.
// tokenIDs and mask must have the same length.
// mask[i] = 0 for masked tokens (system/user), 1 for loss tokens (assistant).
func (w *ShardWriter) WriteSFTDocument(tokenIDs []int, mask []byte) error {
	if len(tokenIDs) != len(mask) {
		return fmt.Errorf("tokenIDs len %d != mask len %d", len(tokenIDs), len(mask))
	}
	if w.maskBuf == nil {
		return fmt.Errorf("shard writer has no mask file (not SFT mode)")
	}

	// Write token IDs.
	nBytes := len(tokenIDs) * 4
	if cap(w.binScratch) < nBytes {
		w.binScratch = make([]byte, nBytes)
	}
	buf := w.binScratch[:nBytes]
	for i, id := range tokenIDs {
		binary.LittleEndian.PutUint32(buf[i*4:], uint32(id))
	}
	if _, err := w.binBuf.Write(buf); err != nil {
		return err
	}

	// Write mask.
	if _, err := w.maskBuf.Write(mask); err != nil {
		return err
	}

	w.binOffset += uint64(len(tokenIDs)) * 4
	w.TokenCount += uint64(len(tokenIDs))
	w.DocCount++

	return nil
}

// Close flushes and closes all files.
func (w *ShardWriter) Close() error {
	var errs []error
	if err := w.binBuf.Flush(); err != nil {
		errs = append(errs, err)
	}
	if w.idxBuf != nil {
		if err := w.idxBuf.Flush(); err != nil {
			errs = append(errs, err)
		}
	}
	if w.maskBuf != nil {
		if err := w.maskBuf.Flush(); err != nil {
			errs = append(errs, err)
		}
	}
	if err := w.binFile.Close(); err != nil {
		errs = append(errs, err)
	}
	if w.idxFile != nil {
		if err := w.idxFile.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if w.maskFile != nil {
		if err := w.maskFile.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}

// Remove deletes the shard's output files.
func (w *ShardWriter) Remove(pathPrefix string) {
	os.Remove(pathPrefix + ".bin")
	os.Remove(pathPrefix + ".idx")
	os.Remove(pathPrefix + ".mask")
}
