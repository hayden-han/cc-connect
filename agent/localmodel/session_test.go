package localmodel

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

// ── Ollama format test server ──────────────────────────────────

func fakeOllamaServer(t *testing.T, chunks []ollamaChunk) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "qwen3:8b", "size": 5200000000},
				{"name": "llama3.1:8b", "size": 4700000000},
			},
		})
	})

	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			w.Write(data)
			w.Write([]byte("\n"))
			if flusher != nil {
				flusher.Flush()
			}
		}
	})

	return httptest.NewServer(mux)
}

// ── OpenAI format test server ──────────────────────────────────

func fakeOpenAIServer(t *testing.T, chunks []openAIChunk, done bool) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"id": "my-local-model"},
				{"id": "another-model"},
			},
		})
	})

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		for _, chunk := range chunks {
			data, _ := json.Marshal(chunk)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if flusher != nil {
				flusher.Flush()
			}
		}
		if done {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			if flusher != nil {
				flusher.Flush()
			}
		}
	})

	return httptest.NewServer(mux)
}

// ── Tests ──────────────────────────────────────────────────────

func TestNewAgent_Ollama(t *testing.T) {
	srv := fakeOllamaServer(t, nil)
	defer srv.Close()

	agent, err := New(map[string]any{"base_url": srv.URL, "model": "qwen3:8b"})
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if agent.Name() != "localmodel" {
		t.Errorf("Name() = %q, want %q", agent.Name(), "localmodel")
	}
	a := agent.(*Agent)
	if a.apiFormat != FormatOllama {
		t.Errorf("apiFormat = %q, want %q (auto-detected)", a.apiFormat, FormatOllama)
	}
}

func TestNewAgent_OpenAI(t *testing.T) {
	srv := fakeOpenAIServer(t, nil, false)
	defer srv.Close()

	agent, err := New(map[string]any{"base_url": srv.URL, "api_format": "openai", "model": "my-model"})
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	a := agent.(*Agent)
	if a.apiFormat != FormatOpenAI {
		t.Errorf("apiFormat = %q, want %q", a.apiFormat, FormatOpenAI)
	}
}

func TestAutoDetect_Ollama(t *testing.T) {
	srv := fakeOllamaServer(t, nil)
	defer srv.Close()

	format := resolveAPIFormat("", srv.URL)
	if format != FormatOllama {
		t.Errorf("auto-detect = %q, want %q", format, FormatOllama)
	}
}

func TestAutoDetect_FallbackOpenAI(t *testing.T) {
	format := resolveAPIFormat("", "http://127.0.0.1:1")
	if format != FormatOpenAI {
		t.Errorf("auto-detect fallback = %q, want %q", format, FormatOpenAI)
	}
}

func TestAvailableModels_Ollama(t *testing.T) {
	srv := fakeOllamaServer(t, nil)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL})
	models := a.(*Agent).AvailableModels(context.Background())
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
}

func TestAvailableModels_OpenAI(t *testing.T) {
	srv := fakeOpenAIServer(t, nil, false)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL, "api_format": "openai"})
	models := a.(*Agent).AvailableModels(context.Background())
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
}

func TestOllamaStream(t *testing.T) {
	chunks := []ollamaChunk{
		{Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"assistant", "Hello"}, Done: false},
		{Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"assistant", " World"}, Done: false},
		{Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"assistant", ""}, Done: true, TotalDuration: 1000000000},
	}

	srv := fakeOllamaServer(t, chunks)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL, "model": "qwen3:8b"})
	sess, _ := a.StartSession(context.Background(), "test-1")
	defer sess.Close()

	sess.Send("Hello", nil, nil)
	texts := drainTexts(t, sess)
	if len(texts) != 2 || texts[0] != "Hello" || texts[1] != " World" {
		t.Errorf("unexpected texts: %v", texts)
	}
}

func TestOpenAIStream(t *testing.T) {
	stop := "stop"
	chunks := []openAIChunk{
		{Choices: []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		}{{Delta: struct {
			Content string `json:"content"`
		}{"Hi"}, FinishReason: nil}}},
		{Choices: []struct {
			Delta struct {
				Content string `json:"content"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		}{{Delta: struct {
			Content string `json:"content"`
		}{" there"}, FinishReason: &stop}}},
	}

	srv := fakeOpenAIServer(t, chunks, false)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL, "api_format": "openai", "model": "test"})
	sess, _ := a.StartSession(context.Background(), "test-2")
	defer sess.Close()

	sess.Send("Hello", nil, nil)
	texts := drainTexts(t, sess)
	if len(texts) != 2 || texts[0] != "Hi" || texts[1] != " there" {
		t.Errorf("unexpected texts: %v", texts)
	}
}

func TestMultiTurn(t *testing.T) {
	chunks := []ollamaChunk{
		{Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"assistant", "Reply"}, Done: false},
		{Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"", ""}, Done: true},
	}

	srv := fakeOllamaServer(t, chunks)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL})
	sess, _ := a.StartSession(context.Background(), "multi")
	defer sess.Close()

	sess.Send("First", nil, nil)
	drainTexts(t, sess)
	sess.Send("Second", nil, nil)
	drainTexts(t, sess)

	ls := sess.(*localSession)
	ls.historyMu.Lock()
	histLen := len(ls.history)
	ls.historyMu.Unlock()

	if histLen != 4 {
		t.Errorf("expected 4 history entries, got %d", histLen)
	}
}

func TestOllamaError(t *testing.T) {
	chunks := []ollamaChunk{
		{Error: "model not found"},
	}

	srv := fakeOllamaServer(t, chunks)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL})
	sess, _ := a.StartSession(context.Background(), "err")
	defer sess.Close()

	sess.Send("Hello", nil, nil)

	timeout := time.After(5 * time.Second)
	select {
	case evt := <-sess.Events():
		if evt.Type != core.EventError {
			t.Errorf("expected EventError, got %v", evt.Type)
		}
	case <-timeout:
		t.Fatal("timeout")
	}
}

// drainTexts reads events until EventResult and returns all text contents.
func drainTexts(t *testing.T, sess core.AgentSession) []string {
	t.Helper()
	var texts []string
	timeout := time.After(5 * time.Second)
	for {
		select {
		case evt := <-sess.Events():
			switch evt.Type {
			case core.EventText:
				texts = append(texts, evt.Content)
			case core.EventResult:
				return texts
			case core.EventError:
				t.Fatalf("unexpected error: %v", evt.Error)
			}
		case <-timeout:
			t.Fatal("timeout draining events")
		}
	}
}
