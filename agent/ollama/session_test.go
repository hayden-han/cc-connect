package ollama

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

// fakeOllamaServer returns a test server that mimics Ollama's /api/chat streaming.
func fakeOllamaServer(t *testing.T, chunks []chatStreamChunk) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	mux.HandleFunc("/api/tags", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "qwen3:8b", "size": 5200000000, "modified_at": "2026-01-01T00:00:00Z"},
				{"name": "llama3.1:8b", "size": 4700000000, "modified_at": "2026-01-01T00:00:00Z"},
			},
		})
	})

	mux.HandleFunc("/api/chat", func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

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

func TestNewAgent(t *testing.T) {
	srv := fakeOllamaServer(t, nil)
	defer srv.Close()

	agent, err := New(map[string]any{
		"base_url": srv.URL,
		"model":    "qwen3:8b",
	})
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}
	if agent.Name() != "ollama" {
		t.Errorf("Name() = %q, want %q", agent.Name(), "ollama")
	}
}

func TestNewAgent_Unreachable(t *testing.T) {
	_, err := New(map[string]any{
		"base_url": "http://127.0.0.1:1", // unreachable
	})
	if err == nil {
		t.Fatal("expected error for unreachable server")
	}
}

func TestAvailableModels(t *testing.T) {
	srv := fakeOllamaServer(t, nil)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL})
	agent := a.(*Agent)

	models := agent.AvailableModels(context.Background())
	if len(models) != 2 {
		t.Fatalf("expected 2 models, got %d", len(models))
	}
}

func TestSessionSendAndStream(t *testing.T) {
	chunks := []chatStreamChunk{
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

	a, err := New(map[string]any{"base_url": srv.URL, "model": "qwen3:8b"})
	if err != nil {
		t.Fatalf("New() failed: %v", err)
	}

	ctx := context.Background()
	sess, err := a.StartSession(ctx, "test-session-1")
	if err != nil {
		t.Fatalf("StartSession() failed: %v", err)
	}
	defer sess.Close()

	if err := sess.Send("Hello", nil, nil); err != nil {
		t.Fatalf("Send() failed: %v", err)
	}

	var texts []string
	var gotResult bool
	timeout := time.After(5 * time.Second)

	for !gotResult {
		select {
		case evt := <-sess.Events():
			switch evt.Type {
			case core.EventText:
				texts = append(texts, evt.Content)
			case core.EventResult:
				gotResult = true
				if evt.SessionID != "test-session-1" {
					t.Errorf("SessionID = %q, want %q", evt.SessionID, "test-session-1")
				}
			case core.EventError:
				t.Fatalf("unexpected error: %v", evt.Error)
			}
		case <-timeout:
			t.Fatal("timeout waiting for events")
		}
	}

	if len(texts) != 2 {
		t.Errorf("expected 2 text events, got %d: %v", len(texts), texts)
	}
}

func TestSessionMultiTurn(t *testing.T) {
	chunks := []chatStreamChunk{
		{Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"assistant", "Response"}, Done: false},
		{Message: struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"assistant", ""}, Done: true},
	}

	srv := fakeOllamaServer(t, chunks)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL, "model": "qwen3:8b"})
	ctx := context.Background()
	sess, _ := a.StartSession(ctx, "multi-turn")
	defer sess.Close()

	// Turn 1
	sess.Send("First message", nil, nil)
	drainUntilResult(t, sess)

	// Turn 2 — history should include previous exchange
	sess.Send("Second message", nil, nil)
	drainUntilResult(t, sess)

	os := sess.(*ollamaSession)
	os.historyMu.Lock()
	histLen := len(os.history)
	os.historyMu.Unlock()

	// Should have: user1, assistant1, user2, assistant2 = 4
	if histLen != 4 {
		t.Errorf("expected 4 history entries, got %d", histLen)
	}
}

func TestSessionError(t *testing.T) {
	chunks := []chatStreamChunk{
		{Error: "model not found"},
	}

	srv := fakeOllamaServer(t, chunks)
	defer srv.Close()

	a, _ := New(map[string]any{"base_url": srv.URL, "model": "nonexistent"})
	ctx := context.Background()
	sess, _ := a.StartSession(ctx, "error-test")
	defer sess.Close()

	sess.Send("Hello", nil, nil)

	timeout := time.After(5 * time.Second)
	select {
	case evt := <-sess.Events():
		if evt.Type != core.EventError {
			t.Errorf("expected EventError, got %v", evt.Type)
		}
	case <-timeout:
		t.Fatal("timeout waiting for error event")
	}
}

func drainUntilResult(t *testing.T, sess core.AgentSession) {
	t.Helper()
	timeout := time.After(5 * time.Second)
	for {
		select {
		case evt := <-sess.Events():
			if evt.Type == core.EventResult {
				return
			}
			if evt.Type == core.EventError {
				t.Fatalf("unexpected error: %v", evt.Error)
			}
		case <-timeout:
			t.Fatal("timeout draining events")
		}
	}
}
