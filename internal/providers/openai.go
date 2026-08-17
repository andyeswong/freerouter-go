// Package providers proxies a chat request to the chosen model's
// OpenAI-compatible endpoint.
package providers

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/andyeswong/freerouter-go/internal/models"
)

// Message accepts content as a string OR an array of {type,text} blocks.
// Hard-won lesson from cc_bridge (2026-06-17): OpenClaw's WhatsApp channel
// sends content as an array while Telegram sends a string; a plain `string`
// field 400s on the array. Flatten to a single string for downstream use.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Role = raw.Role

	// Missing or explicit-null content (e.g. an assistant message that carries
	// only tool_calls). Treat as empty — do NOT error. (Crush/OpenAI clients
	// send assistant tool-call turns with no content field.)
	if len(raw.Content) == 0 || string(raw.Content) == "null" {
		m.Content = ""
		return nil
	}

	// Try string first.
	var s string
	if err := json.Unmarshal(raw.Content, &s); err == nil {
		m.Content = s
		return nil
	}
	// Else array of blocks.
	var blocks []map[string]any
	if err := json.Unmarshal(raw.Content, &blocks); err != nil {
		return fmt.Errorf("content is neither string nor block array: %w", err)
	}
	var parts []string
	for _, blk := range blocks {
		if text, ok := BlockText(blk); ok {
			parts = append(parts, text)
		}
	}
	m.Content = strings.Join(parts, "\n")
	return nil
}

// BlockText pulls readable text out of one OpenAI-style content block. Besides
// the plain {"type":"text","text":...} shape it understands {"type":"file"},
// which is what Coder's aibridge emits when a large paste is attached to the
// turn — the payload rides in file.file_data, normally as a data: URI. Reports
// ok=false when the block carries nothing worth showing the model.
//
// Routing reads this too, not just the proxy: a file block used to measure as
// zero characters, so a huge paste was sized as an empty context.
func BlockText(blk map[string]any) (string, bool) {
	if s, ok := blk["text"].(string); ok && s != "" {
		return s, true
	}
	if t, _ := blk["type"].(string); t != "file" {
		return "", false
	}
	f, _ := blk["file"].(map[string]any)
	if f == nil {
		f = map[string]any{}
	}
	name, _ := f["filename"].(string)
	data, _ := f["file_data"].(string)
	if data == "" {
		// The exact envelope was never captured off the wire (the 400 body named
		// the variant, not its shape), and clients differ — Anthropic-shaped
		// bridges nest the payload under source.data. Fall back to the longest
		// string anywhere in the block so the paste is never silently dropped.
		data = longestString(blk)
	}
	if text, ok := decodeFileData(data); ok {
		if name != "" {
			return name + ":\n" + text, true
		}
		return text, true
	}
	if name != "" {
		// A real binary attachment: name it rather than splice mojibake into
		// the prompt, and rather than drop the turn's only content.
		return "[attachment " + name + " omitted: this model accepts text only]", true
	}
	return "", false
}

// longestString returns the longest string value anywhere inside v, skipping
// the short bookkeeping fields (type, filename, media types). Used only as the
// last resort in BlockText, to find an attachment payload whose key we don't
// recognize.
func longestString(v any) string {
	best := ""
	var walk func(any)
	walk = func(x any) {
		switch t := x.(type) {
		case string:
			if len(t) > len(best) {
				best = t
			}
		case map[string]any:
			for k, val := range t {
				if k == "type" || k == "filename" || k == "media_type" || k == "mime_type" {
					continue
				}
				walk(val)
			}
		case []any:
			for _, val := range t {
				walk(val)
			}
		}
	}
	walk(v)
	return best
}

// decodeFileData turns a file_data payload into text. Accepts a bare string, a
// data: URI (base64 or percent-encoded), and plain base64. Reports false when
// the bytes are not valid UTF-8 — a binary attachment must not reach the model
// as garbage.
func decodeFileData(s string) (string, bool) {
	if s == "" {
		return "", false
	}
	payload := s
	switch {
	case strings.HasPrefix(s, "data:"):
		comma := strings.IndexByte(s, ',')
		if comma < 0 {
			return "", false
		}
		meta, rest := s[len("data:"):comma], s[comma+1:]
		payload = rest
		if strings.Contains(meta, ";base64") {
			b, err := base64.StdEncoding.DecodeString(rest)
			if err != nil {
				return "", false
			}
			payload = string(b)
		} else if unescaped, err := url.QueryUnescape(rest); err == nil {
			payload = unescaped
		}
	default:
		// Unlabelled: it may be base64 or it may already be text. Short words
		// ("hola") happen to be valid base64 and decode to bytes, so only take
		// the decoded form when it reads back as text.
		if b, err := base64.StdEncoding.DecodeString(s); err == nil && utf8.Valid(b) {
			payload = string(b)
		}
	}
	if !utf8.ValidString(payload) {
		return "", false
	}
	return payload, true
}

// flattenContentBlocks rewrites every message's content array so only block
// types the providers actually implement survive. DeepSeek rejects the whole
// request on an unknown variant ("unknown variant `file`, expected `text`",
// seen in prod 2026-07-29 from the Coder agent whenever a large paste was in
// the turn), and the others have no shared superset either — plain text is the
// one shape every endpoint in the vademécum accepts.
func flattenContentBlocks(body map[string]any) {
	msgs, ok := body["messages"].([]any)
	if !ok {
		return
	}
	for _, m := range msgs {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		blocks, ok := msg["content"].([]any)
		if !ok {
			continue // already a plain string
		}
		kept := make([]any, 0, len(blocks))
		for _, b := range blocks {
			blk, ok := b.(map[string]any)
			if !ok {
				kept = append(kept, b)
				continue
			}
			// text and image_url are the two variants providers agree on; pass
			// them through untouched so nothing is re-encoded needlessly.
			if t, _ := blk["type"].(string); t == "text" || t == "image_url" {
				kept = append(kept, b)
				continue
			}
			if text, ok := BlockText(blk); ok {
				kept = append(kept, map[string]any{"type": "text", "text": text})
			}
		}
		if len(kept) == 0 {
			msg["content"] = "" // an empty array 400s on some providers
			continue
		}
		msg["content"] = kept
	}
}

// ChatRequest is the subset of the OpenAI chat schema the router inspects.
// The full original body is proxied verbatim except `model`, which is rewritten
// to the chosen upstream model id.
type ChatRequest struct {
	Model     string          `json:"model"`
	Messages  []Message       `json:"messages"`
	MaxTokens int             `json:"max_tokens"`
	Stream    bool            `json:"stream"`
	Tools     json.RawMessage `json:"tools,omitempty"` // function-calling tool defs (presence steers routing)

	// ResponseFormat carries OpenAI structured-output requests. Only its "type"
	// matters for routing; the field itself is proxied upstream untouched.
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`

	// FreeRouter extensions (optional, ignored by upstream after stripping).
	Tier int `json:"tier,omitempty"`
}

// HasTools reports whether this is a function-calling conversation — either the
// request declares tools, or the history contains a tool result / tool-call turn.
func (r ChatRequest) HasTools() bool {
	if len(r.Tools) > 0 && string(r.Tools) != "null" && string(r.Tools) != "[]" {
		return true
	}
	for _, m := range r.Messages {
		if m.Role == "tool" {
			return true
		}
	}
	return false
}

// RequiresJSONSchema reports whether the caller demands strict structured
// output (response_format:{"type":"json_schema"}). Plain {"type":"json_object"}
// is widely supported and deliberately does NOT narrow routing.
func (r ChatRequest) RequiresJSONSchema() bool {
	if len(r.ResponseFormat) == 0 || string(r.ResponseFormat) == "null" {
		return false
	}
	var rf struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(r.ResponseFormat, &rf) != nil {
		return false
	}
	return rf.Type == "json_schema"
}

// schemaInstruction turns a json_schema response_format into a prompt the model
// can actually obey. Measured 2026-07-28 against every provider in the
// vademécum: DeepSeek answers 400 "This response_format type is unavailable
// now", while Ollama Cloud and cc_bridge answer 200 and IGNORE the field —
// returning prose where the caller expects an object, which is worse than an
// error. None of them implement it. Asked in the prompt, though, they comply.
func schemaInstruction(responseFormat json.RawMessage) string {
	var rf struct {
		JSONSchema struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if json.Unmarshal(responseFormat, &rf) != nil || len(rf.JSONSchema.Schema) == 0 {
		return "Respond with a single valid JSON object and nothing else. No markdown, no code fences, no prose."
	}
	name := rf.JSONSchema.Name
	if name == "" {
		name = "result"
	}
	return "Your entire response must be a single valid JSON object matching this JSON Schema" +
		" (named \"" + name + "\"):\n" + string(rf.JSONSchema.Schema) +
		"\nOutput ONLY that JSON object: no markdown, no code fences, no explanation, no text before or after."
}

// ExtractJSON pulls the first balanced JSON object out of a model's answer, so
// a stray code fence or a sentence of preamble doesn't break a caller that is
// about to json.Parse the content. Brace counting ignores braces inside string
// literals and honors escapes. Returns false when there is no object to find.
func ExtractJSON(s string) (string, bool) {
	start := strings.IndexByte(s, '{')
	if start < 0 {
		return "", false
	}
	depth, inStr, esc := 0, false, false
	for i := start; i < len(s); i++ {
		c := s[i]
		switch {
		case esc:
			esc = false
		case c == '\\' && inStr:
			esc = true
		case c == '"':
			inStr = !inStr
		case inStr:
			// literal content, nothing to count
		case c == '{':
			depth++
		case c == '}':
			depth--
			if depth == 0 {
				return s[start : i+1], true
			}
		}
	}
	return "", false // unterminated (e.g. the answer hit max_tokens)
}

var httpClient = &http.Client{Timeout: 300 * time.Second}

// KeyResolver maps a model's api_key_ref to the actual key. Defaults to env
// lookup; main wires it to check the DB secret store first, then env, so keys
// can be added/rotated at runtime without a restart.
var KeyResolver = func(ref string) string { return os.Getenv(ref) }

// EmulatesJSONSchema reports whether a json_schema request to this model has to
// be served by instruction instead of natively. Handlers use it to decide
// whether the answer needs the JSON extracted before it reaches the caller.
func EmulatesJSONSchema(m models.LlmModel, req ChatRequest) bool {
	return req.RequiresJSONSchema() && !m.JSONSchemaOK
}

// Proxy forwards rawBody to the model's endpoint, rewriting the model id and
// injecting the API key resolved from the model's APIKeyRef env var. Returns
// the upstream response for the handler to stream/relay.
func Proxy(m models.LlmModel, rawBody []byte) (*http.Response, error) {
	// Rewrite the "model" field to the upstream id; strip router-only fields.
	var body map[string]any
	if err := json.Unmarshal(rawBody, &body); err != nil {
		return nil, err
	}
	body["model"] = m.ModelID
	delete(body, "tier")
	delete(body, "requires_mcp")

	// Reduce content blocks to the variants every provider implements, before
	// anything else appends to messages.
	flattenContentBlocks(body)

	// Structured output for a model that doesn't implement it: carry the schema
	// as an instruction and drop the field, which would otherwise earn a 400
	// from DeepSeek or be silently ignored elsewhere. Appended LAST for recency,
	// the same reason cc_bridge puts its router directive at the end.
	if rf, ok := body["response_format"].(map[string]any); ok && !m.JSONSchemaOK {
		if t, _ := rf["type"].(string); t == "json_schema" {
			raw, _ := json.Marshal(rf)
			delete(body, "response_format")
			if msgs, ok := body["messages"].([]any); ok {
				body["messages"] = append(msgs, map[string]any{
					"role": "system", "content": schemaInstruction(raw),
				})
			}
		}
	}

	// Inject the model's custom system prompt at the front of the messages, so
	// it steers every request routed to this model.
	if m.CustomSystemPrompt != "" {
		if msgs, ok := body["messages"].([]any); ok {
			sys := map[string]any{"role": "system", "content": m.CustomSystemPrompt}
			body["messages"] = append([]any{sys}, msgs...)
		}
	}

	// Force plan-only mode (cc_bridge no_exec): return tool_calls to the parent
	// harness instead of executing. Travels in the body so it reaches cc_bridge.
	if m.ForceNoExec {
		body["no_exec"] = true
	}

	// Ask the upstream to include token usage in the final stream chunk so we
	// can bill streamed requests too (OpenAI-compatible stream_options).
	if s, _ := body["stream"].(bool); s {
		body["stream_options"] = map[string]any{"include_usage": true}
	}

	out, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	url := strings.TrimRight(m.APIBaseURL, "/") + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(out))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key := KeyResolver(m.APIKeyRef); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	return httpClient.Do(req)
}

// ExtractPrompt returns the last user message and the concatenated system
// prompt, for the router to classify.
func ExtractPrompt(msgs []Message) (user, system string) {
	var sys []string
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" && user == "" {
			user = msgs[i].Content
		}
		if msgs[i].Role == "system" {
			sys = append([]string{msgs[i].Content}, sys...)
		}
	}
	return user, strings.Join(sys, "\n")
}

// ContextChars sums characters across ALL messages (plus a small per-message
// overhead for role/formatting tokens). This is the real size of the request —
// used to size context for routing, unlike the last-prompt-only classifier.
func ContextChars(msgs []Message) int {
	n := 0
	for _, m := range msgs {
		n += len(m.Content) + 8
	}
	return n
}

// Drain is a small helper to fully read+close a response body.
func Drain(r io.ReadCloser) { _, _ = io.Copy(io.Discard, r); _ = r.Close() }

// Usage holds the token counts reported by the upstream.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ParseUsage extracts token usage from a full response body. Handles both a
// plain JSON completion and an SSE stream (scans `data:` lines, keeps the last
// one carrying a usage object — that's the include_usage final chunk).
func ParseUsage(body []byte) Usage {
	var u Usage

	// Non-stream: a single JSON object with a top-level "usage".
	var obj struct {
		Usage Usage `json:"usage"`
	}
	if json.Unmarshal(body, &obj) == nil && obj.Usage.TotalTokens > 0 {
		return obj.Usage
	}

	// Stream: walk SSE lines.
	for _, line := range strings.Split(string(body), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var chunk struct {
			Usage *Usage `json:"usage"`
		}
		if json.Unmarshal([]byte(payload), &chunk) == nil && chunk.Usage != nil && chunk.Usage.TotalTokens > 0 {
			u = *chunk.Usage
		}
	}
	return u
}
