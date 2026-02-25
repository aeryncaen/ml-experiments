// Package ingest reads parquet files and writes raw JSONL shards.
package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
	"github.com/apache/arrow-go/v18/parquet"
	"github.com/apache/arrow-go/v18/parquet/file"
	"github.com/apache/arrow-go/v18/parquet/pqarrow"
)

// ParquetReader reads records from a parquet file, yielding each row as a
// map[string]any suitable for JSON serialization.
type ParquetReader struct {
	path   string
	reader *file.Reader
	arrow  *pqarrow.FileReader
}

// OpenParquet opens a parquet file for reading.
func OpenParquet(path string) (*ParquetReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open parquet %s: %w", path, err)
	}

	props := parquet.ReaderProperties{BufferSize: 1 << 20}
	pqReader, err := file.NewParquetReader(f, file.WithReadProps(&props))
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("open parquet reader %s: %w", path, err)
	}

	arrowReader, err := pqarrow.NewFileReader(pqReader, pqarrow.ArrowReadProperties{
		Parallel:  true,
		BatchSize: 4096,
	}, memory.DefaultAllocator)
	if err != nil {
		pqReader.Close()
		return nil, fmt.Errorf("create arrow reader %s: %w", path, err)
	}

	return &ParquetReader{
		path:   path,
		reader: pqReader,
		arrow:  arrowReader,
	}, nil
}

// Schema returns the Arrow schema of the parquet file.
func (r *ParquetReader) Schema() (*arrow.Schema, error) {
	return r.arrow.Schema()
}

// NumRows returns the total number of rows in the parquet file.
func (r *ParquetReader) NumRows() int64 {
	return r.reader.NumRows()
}

// ColumnNames returns all column names in the schema.
func (r *ParquetReader) ColumnNames() []string {
	schema, err := r.Schema()
	if err != nil {
		return nil
	}
	names := make([]string, schema.NumFields())
	for i, f := range schema.Fields() {
		names[i] = f.Name
	}
	return names
}

// ReadAll reads all rows from the parquet file, yielding each as a JSON-serializable map.
// It reads row group by row group to keep memory bounded.
func (r *ParquetReader) ReadAll(fn func(record map[string]any) error) error {
	ctx := context.Background()
	for rg := 0; rg < r.reader.NumRowGroups(); rg++ {
		table, err := r.arrow.ReadRowGroups(ctx, nil, []int{rg})
		if err != nil {
			return fmt.Errorf("read row group %d of %s: %w", rg, r.path, err)
		}

		reader := array.NewTableReader(table, 4096)
		for reader.Next() {
			rec := reader.Record()
			schema := rec.Schema()
			nRows := int(rec.NumRows())
			nCols := int(rec.NumCols())

			for row := 0; row < nRows; row++ {
				record := make(map[string]any, nCols)
				for col := 0; col < nCols; col++ {
					colName := schema.Field(col).Name
					arr := rec.Column(col)
					record[colName] = extractValue(arr, row)
				}
				if err := fn(record); err != nil {
					reader.Release()
					table.Release()
					return err
				}
			}
		}
		reader.Release()
		table.Release()
	}
	return nil
}

// Close closes the parquet reader.
func (r *ParquetReader) Close() error {
	return r.reader.Close()
}

// extractValue extracts a Go value from an Arrow array at the given index.
func extractValue(arr arrow.Array, idx int) any {
	if arr.IsNull(idx) {
		return nil
	}

	switch a := arr.(type) {
	case *array.String:
		return a.Value(idx)
	case *array.LargeString:
		return a.Value(idx)
	case *array.Int8:
		return a.Value(idx)
	case *array.Int16:
		return a.Value(idx)
	case *array.Int32:
		return a.Value(idx)
	case *array.Int64:
		return a.Value(idx)
	case *array.Uint8:
		return a.Value(idx)
	case *array.Uint16:
		return a.Value(idx)
	case *array.Uint32:
		return a.Value(idx)
	case *array.Uint64:
		return a.Value(idx)
	case *array.Float16:
		return a.Value(idx).Float32()
	case *array.Float32:
		return a.Value(idx)
	case *array.Float64:
		return a.Value(idx)
	case *array.Boolean:
		return a.Value(idx)
	case *array.Binary:
		return a.Value(idx)
	case *array.LargeBinary:
		return a.Value(idx)
	case *array.List:
		start, end := a.ValueOffsets(idx)
		child := a.ListValues()
		vals := make([]any, end-start)
		for i := start; i < end; i++ {
			vals[i-start] = extractValue(child, int(i))
		}
		return vals
	case *array.Struct:
		dt := a.DataType().(*arrow.StructType)
		m := make(map[string]any, a.NumField())
		for i := 0; i < a.NumField(); i++ {
			m[dt.Field(i).Name] = extractValue(a.Field(i), idx)
		}
		return m
	default:
		// Fallback: use string representation.
		return fmt.Sprintf("%v", arr.ValueStr(idx))
	}
}

// WriteJSONL writes records as JSONL to a writer.
func WriteJSONL(w io.Writer, records []map[string]any) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	for _, rec := range records {
		if err := enc.Encode(rec); err != nil {
			return err
		}
	}
	return nil
}
