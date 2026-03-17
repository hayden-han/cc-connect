package ollama

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

// chatMessage represents a message in the Ollama chat API.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatRequest is the request body for /api/chat.
type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

// chatStreamChunk is a single chunk from the streaming response.
type chatStreamChunk struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done         bool   `json:"done"`
	DoneReason   string `json:"done_reason,omitempty"`
	TotalDuration int64 `json:"total_duration,omitempty"`
	Error        string `json:"error,omitempty"`
}

// ollamaSession manages a multi-turn conversation with an Ollama model.
// Conversation history is maintained in memory, since Ollama's API is stateless.
type ollamaSession struct {
	baseURL   string
	model     string
	sessionID string
	timeout   time.Duration
	events    chan core.Event
	history   []chatMessage
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	alive     atomic.Bool
	historyMu sync.Mutex
}

func newSession(ctx context.Context, baseURL, model, sessionID string, timeout time.Duration) (*ollamaSession, error) {
	sessionCtx, cancel := context.WithCancel(ctx)

	s := &ollamaSession{
		baseURL:   baseURL,
		model:     model,
		sessionID: sessionID,
		timeout:   timeout,
		events:    make(chan core.Event, 64),
		ctx:       sessionCtx,
		cancel:    cancel,
	}
	s.alive.Store(true)

	return s, nil
}

func (s *ollamaSession) Send(prompt string, images []core.ImageAttachment, files []core.FileAttachment) error {
	if !s.alive.Load() {
		return fmt.Errorf("session is closed")
	}

	// Append user message to history
	s.historyMu.Lock()
	s.history = append(s.history, chatMessage{Role: "user", Content: prompt})
	messages := make([]chatMessage, len(s.history))
	copy(messages, s.history)
	s.historyMu.Unlock()

	reqBody := chatRequest{
		Model:    s.model,
		Messages: messages,
		Stream:   true,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("ollama: marshal request: %w", err)
	}

	var reqCtx context.Context
	var reqCancel context.CancelFunc
	if s.timeout > 0 {
		reqCtx, reqCancel = context.WithTimeout(s.ctx, s.timeout)
	} else {
		reqCtx, reqCancel = context.WithCancel(s.ctx)
	}

	req, err := http.NewRequestWithContext(reqCtx, "POST", s.baseURL+"/api/chat", bytes.NewReader(bodyBytes))
	if err != nil {
		reqCancel()
		return fmt.Errorf("ollama: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		reqCancel()
		return fmt.Errorf("ollama: request failed: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		reqCancel()
		return fmt.Errorf("ollama: API returned %d", resp.StatusCode)
	}

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer reqCancel()
		defer resp.Body.Close()
		s.readStream(resp)
	}()

	return nil
}

func (s *ollamaSession) readStream(resp *http.Response) {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1*1024*1024)

	var fullContent strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			slog.Debug("ollama: non-JSON line", "line", line)
			continue
		}

		if chunk.Error != "" {
			evt := core.Event{Type: core.EventError, Error: fmt.Errorf("ollama: %s", chunk.Error)}
			select {
			case s.events <- evt:
			case <-s.ctx.Done():
				return
			}
			return
		}

		if chunk.Message.Content != "" {
			fullContent.WriteString(chunk.Message.Content)

			evt := core.Event{Type: core.EventText, Content: chunk.Message.Content}
			select {
			case s.events <- evt:
			case <-s.ctx.Done():
				return
			}
		}

		if chunk.Done {
			// Append assistant response to history
			s.historyMu.Lock()
			s.history = append(s.history, chatMessage{Role: "assistant", Content: fullContent.String()})
			s.historyMu.Unlock()

			var durationInfo string
			if chunk.TotalDuration > 0 {
				dur := time.Duration(chunk.TotalDuration)
				durationInfo = fmt.Sprintf(" (%.1fs)", dur.Seconds())
			}

			slog.Debug("ollama: response complete", "model", s.model, "duration", durationInfo)

			evt := core.Event{Type: core.EventResult, SessionID: s.sessionID, Done: true}
			select {
			case s.events <- evt:
			case <-s.ctx.Done():
				return
			}
			return
		}
	}

	if err := scanner.Err(); err != nil {
		slog.Error("ollama: scanner error", "error", err)
		evt := core.Event{Type: core.EventError, Error: fmt.Errorf("ollama: read stream: %w", err)}
		select {
		case s.events <- evt:
		case <-s.ctx.Done():
		}
	}
}

// RespondPermission is a no-op — Ollama does not have a permission system.
func (s *ollamaSession) RespondPermission(_ string, _ core.PermissionResult) error {
	return nil
}

func (s *ollamaSession) Events() <-chan core.Event {
	return s.events
}

func (s *ollamaSession) CurrentSessionID() string {
	return s.sessionID
}

func (s *ollamaSession) Alive() bool {
	return s.alive.Load()
}

func (s *ollamaSession) Close() error {
	s.alive.Store(false)
	s.cancel()
	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		slog.Warn("ollama: close timed out")
	}
	close(s.events)
	return nil
}
