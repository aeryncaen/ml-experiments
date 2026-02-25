// Package process runs per-dataset Lua scripts over raw JSONL records.
package process

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode/utf8"

	lua "github.com/yuin/gopher-lua"
)

// SFTSegment is a single segment in a structured SFT conversation.
type SFTSegment struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Loss    bool   `json:"loss"`
}

// ProcessedRecord is the output of a Lua extract() call.
type ProcessedRecord struct {
	PTText      string       `json:"pt_text"`
	SFTSegments []SFTSegment `json:"sft_segments,omitempty"`
	Route       string       `json:"route"` // "train", "val", or "skip"
}

// LuaEnv wraps a gopher-lua VM with sandboxing and Go↔Lua bindings.
type LuaEnv struct {
	vm         *lua.LState
	extractFn  *lua.LFunction
	scriptPath string
}

// NewLuaEnv creates a sandboxed Lua VM and loads the given script.
func NewLuaEnv(scriptPath string) (*LuaEnv, error) {
	vm := lua.NewState(lua.Options{
		SkipOpenLibs: true,
	})

	// Open only safe libraries.
	for _, lib := range []struct {
		name string
		fn   lua.LGFunction
	}{
		{lua.BaseLibName, lua.OpenBase},
		{lua.TabLibName, lua.OpenTable},
		{lua.StringLibName, lua.OpenString},
		{lua.MathLibName, lua.OpenMath},
	} {
		vm.Push(vm.NewFunction(lib.fn))
		vm.Push(lua.LString(lib.name))
		vm.Call(1, 0)
	}

	// Remove unsafe functions from base lib.
	for _, name := range []string{"dofile", "loadfile", "load", "loadstring", "print"} {
		vm.SetGlobal(name, lua.LNil)
	}

	// Register Go bindings.
	registerBindings(vm)

	// Load the script.
	data, err := os.ReadFile(scriptPath)
	if err != nil {
		vm.Close()
		return nil, fmt.Errorf("read script %s: %w", scriptPath, err)
	}

	fn, err := vm.LoadString(string(data))
	if err != nil {
		vm.Close()
		return nil, fmt.Errorf("compile script %s: %w", scriptPath, err)
	}

	vm.Push(fn)
	if err := vm.PCall(0, lua.MultRet, nil); err != nil {
		vm.Close()
		return nil, fmt.Errorf("exec script %s: %w", scriptPath, err)
	}

	// Get the extract function.
	extractVal := vm.GetGlobal("extract")
	extractFn, ok := extractVal.(*lua.LFunction)
	if !ok {
		vm.Close()
		return nil, fmt.Errorf("script %s: global 'extract' is not a function (got %T)", scriptPath, extractVal)
	}

	return &LuaEnv{
		vm:         vm,
		extractFn:  extractFn,
		scriptPath: scriptPath,
	}, nil
}

// Close shuts down the Lua VM.
func (env *LuaEnv) Close() {
	env.vm.Close()
}

// Extract calls the Lua extract(record) function and returns the processed result.
// Returns nil if the Lua function returns nil (skip this record).
func (env *LuaEnv) Extract(record map[string]any) (*ProcessedRecord, error) {
	// Convert Go map → Lua table.
	tbl := goMapToLuaTable(env.vm, record)

	// Call extract(record).
	if err := env.vm.CallByParam(lua.P{
		Fn:      env.extractFn,
		NRet:    1,
		Protect: true,
	}, tbl); err != nil {
		return nil, fmt.Errorf("lua extract(): %w", err)
	}

	// Get return value.
	ret := env.vm.Get(-1)
	env.vm.Pop(1)

	if ret == lua.LNil {
		return nil, nil // skip record
	}

	resultTbl, ok := ret.(*lua.LTable)
	if !ok {
		return nil, fmt.Errorf("lua extract() returned %T, expected table or nil", ret)
	}

	return luaTableToResult(resultTbl)
}

func registerBindings(vm *lua.LState) {
	// log(msg) — print to stderr
	vm.SetGlobal("log", vm.NewFunction(func(L *lua.LState) int {
		msg := L.CheckString(1)
		fmt.Fprintf(os.Stderr, "[lua] %s\n", msg)
		return 0
	}))

	// utf8_len(s) — fast UTF-8 character count
	vm.SetGlobal("utf8_len", vm.NewFunction(func(L *lua.LState) int {
		s := L.CheckString(1)
		L.Push(lua.LNumber(utf8.RuneCountInString(s)))
		return 1
	}))

	// trim(s) — whitespace trim
	vm.SetGlobal("trim", vm.NewFunction(func(L *lua.LState) int {
		s := L.CheckString(1)
		L.Push(lua.LString(strings.TrimSpace(s)))
		return 1
	}))

	// json_encode(t) — serialize table to JSON
	vm.SetGlobal("json_encode", vm.NewFunction(func(L *lua.LState) int {
		val := L.CheckTable(1)
		goVal := luaValueToGo(val)
		data, err := json.Marshal(goVal)
		if err != nil {
			L.ArgError(1, "cannot JSON encode: "+err.Error())
			return 0
		}
		L.Push(lua.LString(string(data)))
		return 1
	}))

	// sha256(s) — hash for dedup fingerprinting
	vm.SetGlobal("sha256", vm.NewFunction(func(L *lua.LState) int {
		s := L.CheckString(1)
		h := sha256.Sum256([]byte(s))
		L.Push(lua.LString(hex.EncodeToString(h[:])))
		return 1
	}))

	// has_field(t, k) — nil-safe field check
	vm.SetGlobal("has_field", vm.NewFunction(func(L *lua.LState) int {
		tbl := L.CheckTable(1)
		key := L.CheckString(2)
		val := tbl.RawGetString(key)
		L.Push(lua.LBool(val != lua.LNil))
		return 1
	}))
}

// goMapToLuaTable converts a Go map to a Lua table.
func goMapToLuaTable(L *lua.LState, m map[string]any) *lua.LTable {
	tbl := L.NewTable()
	for k, v := range m {
		tbl.RawSetString(k, goValueToLua(L, v))
	}
	return tbl
}

// goValueToLua converts a Go value to a Lua value.
func goValueToLua(L *lua.LState, v any) lua.LValue {
	if v == nil {
		return lua.LNil
	}
	switch val := v.(type) {
	case string:
		return lua.LString(val)
	case float64:
		return lua.LNumber(val)
	case float32:
		return lua.LNumber(float64(val))
	case int:
		return lua.LNumber(float64(val))
	case int8:
		return lua.LNumber(float64(val))
	case int16:
		return lua.LNumber(float64(val))
	case int32:
		return lua.LNumber(float64(val))
	case int64:
		return lua.LNumber(float64(val))
	case uint8:
		return lua.LNumber(float64(val))
	case uint16:
		return lua.LNumber(float64(val))
	case uint32:
		return lua.LNumber(float64(val))
	case uint64:
		return lua.LNumber(float64(val))
	case bool:
		return lua.LBool(val)
	case []any:
		tbl := L.NewTable()
		for i, item := range val {
			tbl.RawSetInt(i+1, goValueToLua(L, item))
		}
		return tbl
	case map[string]any:
		return goMapToLuaTable(L, val)
	case []byte:
		return lua.LString(string(val))
	default:
		// Fallback: use JSON encoding.
		data, err := json.Marshal(val)
		if err != nil {
			return lua.LString(fmt.Sprintf("%v", val))
		}
		return lua.LString(string(data))
	}
}

// luaValueToGo converts a Lua value to a Go value.
func luaValueToGo(v lua.LValue) any {
	switch val := v.(type) {
	case *lua.LNilType:
		return nil
	case lua.LBool:
		return bool(val)
	case lua.LNumber:
		return float64(val)
	case lua.LString:
		return string(val)
	case *lua.LTable:
		// Detect if it's an array or a map.
		maxN := val.MaxN()
		if maxN > 0 {
			arr := make([]any, 0, maxN)
			for i := 1; i <= maxN; i++ {
				arr = append(arr, luaValueToGo(val.RawGetInt(i)))
			}
			return arr
		}
		m := make(map[string]any)
		val.ForEach(func(key, value lua.LValue) {
			if ks, ok := key.(lua.LString); ok {
				m[string(ks)] = luaValueToGo(value)
			}
		})
		return m
	default:
		return fmt.Sprintf("%v", v)
	}
}

// luaTableToResult converts a Lua table returned by extract() into a ProcessedRecord.
func luaTableToResult(tbl *lua.LTable) (*ProcessedRecord, error) {
	rec := &ProcessedRecord{}

	// pt_text (required)
	ptVal := tbl.RawGetString("pt_text")
	if ptVal != lua.LNil {
		if s, ok := ptVal.(lua.LString); ok {
			rec.PTText = string(s)
		}
	}

	// route (required)
	routeVal := tbl.RawGetString("route")
	if routeVal != lua.LNil {
		if s, ok := routeVal.(lua.LString); ok {
			rec.Route = string(s)
		}
	}
	if rec.Route == "" {
		rec.Route = "train"
	}

	// sft_segments (optional)
	sftVal := tbl.RawGetString("sft_segments")
	if sftVal != lua.LNil {
		if sftTbl, ok := sftVal.(*lua.LTable); ok {
			maxN := sftTbl.MaxN()
			rec.SFTSegments = make([]SFTSegment, 0, maxN)
			for i := 1; i <= maxN; i++ {
				segVal := sftTbl.RawGetInt(i)
				segTbl, ok := segVal.(*lua.LTable)
				if !ok {
					continue
				}
				seg := SFTSegment{}
				if v, ok := segTbl.RawGetString("role").(lua.LString); ok {
					seg.Role = string(v)
				}
				if v, ok := segTbl.RawGetString("content").(lua.LString); ok {
					seg.Content = string(v)
				}
				if v, ok := segTbl.RawGetString("loss").(lua.LBool); ok {
					seg.Loss = bool(v)
				}
				rec.SFTSegments = append(rec.SFTSegments, seg)
			}
		}
	}

	return rec, nil
}
