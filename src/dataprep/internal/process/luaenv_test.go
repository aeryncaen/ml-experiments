package process

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTempScript(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "test.lua")
	if err := os.WriteFile(p, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBasicExtract(t *testing.T) {
	script := `
function extract(record)
    return {
        pt_text = record.text,
        route = "train",
    }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": "hello world"})
	if err != nil {
		t.Fatal(err)
	}
	if rec == nil {
		t.Fatal("expected non-nil result")
	}
	if rec.PTText != "hello world" {
		t.Errorf("pt_text = %q", rec.PTText)
	}
	if rec.Route != "train" {
		t.Errorf("route = %q", rec.Route)
	}
}

func TestExtractReturnNil(t *testing.T) {
	script := `
function extract(record)
    return nil
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": ""})
	if err != nil {
		t.Fatal(err)
	}
	if rec != nil {
		t.Errorf("expected nil result, got %+v", rec)
	}
}

func TestExtractSFTSegments(t *testing.T) {
	script := `
function extract(record)
    return {
        pt_text = record.text,
        route = "train",
        sft_segments = {
            { role = "system", content = "You are helpful.", loss = false },
            { role = "user", content = record.question, loss = false },
            { role = "assistant", content = record.answer, loss = true },
        },
    }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{
		"text":     "combined text",
		"question": "What is 2+2?",
		"answer":   "4",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.SFTSegments) != 3 {
		t.Fatalf("sft_segments len = %d", len(rec.SFTSegments))
	}
	if rec.SFTSegments[0].Role != "system" || rec.SFTSegments[0].Loss {
		t.Errorf("seg[0] = %+v", rec.SFTSegments[0])
	}
	if rec.SFTSegments[1].Content != "What is 2+2?" {
		t.Errorf("seg[1].content = %q", rec.SFTSegments[1].Content)
	}
	if rec.SFTSegments[2].Role != "assistant" || !rec.SFTSegments[2].Loss {
		t.Errorf("seg[2] = %+v", rec.SFTSegments[2])
	}
}

func TestDefaultRoute(t *testing.T) {
	script := `
function extract(record)
    return { pt_text = record.text }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if rec.Route != "train" {
		t.Errorf("default route = %q, want 'train'", rec.Route)
	}
}

func TestBindings_Utf8Len(t *testing.T) {
	script := `
function extract(record)
    local len = utf8_len(record.text)
    return { pt_text = tostring(len), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	if rec.PTText != "5" {
		t.Errorf("utf8_len('hello') = %q", rec.PTText)
	}

	// Multi-byte chars
	rec, err = env.Extract(map[string]any{"text": "日本語"})
	if err != nil {
		t.Fatal(err)
	}
	if rec.PTText != "3" {
		t.Errorf("utf8_len('日本語') = %q", rec.PTText)
	}
}

func TestBindings_Trim(t *testing.T) {
	script := `
function extract(record)
    return { pt_text = trim(record.text), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": "  hello  \n"})
	if err != nil {
		t.Fatal(err)
	}
	if rec.PTText != "hello" {
		t.Errorf("trim result = %q", rec.PTText)
	}
}

func TestBindings_Sha256(t *testing.T) {
	script := `
function extract(record)
    return { pt_text = sha256(record.text), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	// sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
	want := "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
	if rec.PTText != want {
		t.Errorf("sha256('hello') = %q, want %q", rec.PTText, want)
	}
}

func TestBindings_HasField(t *testing.T) {
	script := `
function extract(record)
    local has = has_field(record, "text")
    local missing = has_field(record, "nonexistent")
    return { pt_text = tostring(has) .. "," .. tostring(missing), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if rec.PTText != "true,false" {
		t.Errorf("has_field result = %q", rec.PTText)
	}
}

func TestBindings_JsonEncode(t *testing.T) {
	script := `
function extract(record)
    local tbl = { a = 1, b = "two" }
    return { pt_text = json_encode(tbl), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{"text": ""})
	if err != nil {
		t.Fatal(err)
	}
	// JSON output should contain both keys
	if rec.PTText == "" {
		t.Error("json_encode returned empty")
	}
}

func TestSandboxing(t *testing.T) {
	// Unsafe functions should be nil
	script := `
function extract(record)
    if dofile ~= nil then error("dofile should be nil") end
    if loadfile ~= nil then error("loadfile should be nil") end
    if load ~= nil then error("load should be nil") end
    if print ~= nil then error("print should be nil") end
    return { pt_text = "safe", route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if rec.PTText != "safe" {
		t.Errorf("pt_text = %q", rec.PTText)
	}
}

func TestNumericTypes(t *testing.T) {
	script := `
function extract(record)
    return { pt_text = tostring(record.int_val + record.float_val), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{
		"int_val":   int64(10),
		"float_val": float64(2.5),
	})
	if err != nil {
		t.Fatal(err)
	}
	if rec.PTText != "12.5" {
		t.Errorf("pt_text = %q", rec.PTText)
	}
}

func TestNestedData(t *testing.T) {
	script := `
function extract(record)
    local items = record.tags
    local n = #items
    return { pt_text = tostring(n), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	rec, err := env.Extract(map[string]any{
		"tags": []any{"a", "b", "c"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if rec.PTText != "3" {
		t.Errorf("array len = %q", rec.PTText)
	}
}

func TestScriptError(t *testing.T) {
	script := `
function extract(record)
    error("intentional error")
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	_, err = env.Extract(map[string]any{})
	if err == nil {
		t.Fatal("expected error from script")
	}
}

func TestMissingExtractFunction(t *testing.T) {
	script := `-- no extract function defined`
	_, err := NewLuaEnv(writeTempScript(t, script))
	if err == nil {
		t.Fatal("expected error for missing extract function")
	}
}

func TestInvalidScript(t *testing.T) {
	_, err := NewLuaEnv(writeTempScript(t, "this is not valid lua!!!"))
	if err == nil {
		t.Fatal("expected error for invalid Lua syntax")
	}
}

func TestMissingScriptFile(t *testing.T) {
	_, err := NewLuaEnv("/nonexistent/script.lua")
	if err == nil {
		t.Fatal("expected error for missing script file")
	}
}

func TestMultipleExtracts(t *testing.T) {
	script := `
local count = 0
function extract(record)
    count = count + 1
    return { pt_text = record.text .. ":" .. tostring(count), route = "train" }
end
`
	env, err := NewLuaEnv(writeTempScript(t, script))
	if err != nil {
		t.Fatal(err)
	}
	defer env.Close()

	for i := 1; i <= 5; i++ {
		rec, err := env.Extract(map[string]any{"text": "doc"})
		if err != nil {
			t.Fatal(err)
		}
		want := "doc:" + string(rune('0'+i))
		if rec.PTText != want {
			t.Errorf("call %d: pt_text = %q, want %q", i, rec.PTText, want)
		}
	}
}
