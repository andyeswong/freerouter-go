package providers

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/andyeswong/freerouter-go/internal/models"
)

func TestExtractJSON(t *testing.T) {
	cases := []struct {
		name, in, want string
		found          bool
	}{
		{"bare object", `{"a":1}`, `{"a":1}`, true},
		{"code fence", "```json\n{\"a\":1}\n```", `{"a":1}`, true},
		{"prose before and after", `Sure! {"a":1} hope that helps`, `{"a":1}`, true},
		{"nested", `{"a":{"b":[1,2]},"c":3}`, `{"a":{"b":[1,2]},"c":3}`, true},
		{"brace inside string", `{"a":"}{ not a brace"}`, `{"a":"}{ not a brace"}`, true},
		{"escaped quote then brace", `{"a":"say \"hi\"","b":{"c":1}}`, `{"a":"say \"hi\"","b":{"c":1}}`, true},
		{"no object", `there is no json here`, "", false},
		{"unterminated (hit max_tokens)", `{"a":1,"b":`, "", false},
	}
	for _, tc := range cases {
		got, found := ExtractJSON(tc.in)
		if found != tc.found || got != tc.want {
			t.Errorf("%s: ExtractJSON(%q) = (%q,%v), want (%q,%v)", tc.name, tc.in, got, found, tc.want, tc.found)
		}
	}
}

const schemaBody = `{"model":"auto","messages":[{"role":"user","content":"hi"}],
  "response_format":{"type":"json_schema","json_schema":{"name":"tarea","schema":{"type":"object","properties":{"t":{"type":"string"}}}}}}`

// A model that doesn't implement json_schema must receive the schema as an
// instruction and NOT the response_format field — DeepSeek 400s on it and the
// others ignore it silently.
func TestProxyEmulatesSchemaForIncapableModel(t *testing.T) {
	var seen map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &seen)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"{}"}}]}`))
	}))
	defer srv.Close()

	m := models.LlmModel{ModelID: "m1", APIBaseURL: srv.URL, JSONSchemaOK: false}
	resp, err := Proxy(m, []byte(schemaBody))
	if err != nil {
		t.Fatalf("proxy: %v", err)
	}
	Drain(resp.Body)

	if _, present := seen["response_format"]; present {
		t.Error("response_format reached an incapable provider; it must be stripped")
	}
	msgs, _ := seen["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %d, want 2 (original + instruction)", len(msgs))
	}
	last, _ := msgs[len(msgs)-1].(map[string]any)
	content, _ := last["content"].(string)
	if !strings.Contains(content, "JSON Schema") || !strings.Contains(content, `"t"`) {
		t.Errorf("instruction does not carry the schema: %q", content)
	}
	if role, _ := last["role"].(string); role != "system" {
		t.Errorf("instruction role = %q, want system", role)
	}
}

// A natively capable model must get the field untouched and no instruction.
func TestProxyLeavesSchemaAloneForCapableModel(t *testing.T) {
	var seen map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &seen)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	m := models.LlmModel{ModelID: "m1", APIBaseURL: srv.URL, JSONSchemaOK: true}
	resp, err := Proxy(m, []byte(schemaBody))
	if err != nil {
		t.Fatalf("proxy: %v", err)
	}
	Drain(resp.Body)

	if _, present := seen["response_format"]; !present {
		t.Error("a capable model must still receive response_format")
	}
	if msgs, _ := seen["messages"].([]any); len(msgs) != 1 {
		t.Errorf("messages = %d, want 1 (no instruction appended)", len(msgs))
	}
}

func TestEmulatesJSONSchema(t *testing.T) {
	var req ChatRequest
	if err := json.Unmarshal([]byte(schemaBody), &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !EmulatesJSONSchema(models.LlmModel{JSONSchemaOK: false}, req) {
		t.Error("incapable model + json_schema request should emulate")
	}
	if EmulatesJSONSchema(models.LlmModel{JSONSchemaOK: true}, req) {
		t.Error("capable model must not emulate")
	}
	var plain ChatRequest
	_ = json.Unmarshal([]byte(`{"messages":[]}`), &plain)
	if EmulatesJSONSchema(models.LlmModel{}, plain) {
		t.Error("a request without response_format must not emulate")
	}
}
