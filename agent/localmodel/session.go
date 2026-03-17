package localmodel

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

// chatMessage represents a message in both Ollama and OpenAI chat APIs.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// localSession manages a multi-turn conversation with a local LLM server.
// Conversation history is maintained in memory since local LLM APIs are stateless.
type localSession struct {
	baseURL   string
	model     string
	sessionID string
	apiFormat APIFormat
	apiKey    string
	timeout   time.Duration
	events    chan core.Event
	history   []chatMessage
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	alive     atomic.Bool
	historyMu sync.Mutex
}

func newSession(ctx context.Context, baseURL, model, sessionID string, timeout time.Duration, apiFormat APIFormat, apiKey string) (*localSession, error) {
	sessionCtx, cancel := context.WithCancel(ctx)

	s := &localSession{
		baseURL:   baseURL,
		model:     model,
		sessionID: sessionID,
		apiFormat: apiFormat,
		apiKey:    apiKey,
		timeout:   timeout,
		events:    make(chan core.Event, 64),
		ctx:       sessionCtx,
		cancel:    cancel,
	}
	s.alive.Store(true)

	return s, nil
}

func (s *localSession) Send(prompt string, images []core.ImageAttachment, files []core.FileAttachment) error {
	if !s.alive.Load() {
		return fmt.Errorf("session is closed")
	}

	s.historyMu.Lock()
	s.history = append(s.history, chatMessage{Role: "user", Content: prompt})
	messages := make([]chatMessage, len(s.history))
	copy(messages, s.history)
	s.historyMu.Unlock()

	var reqCtx context.Context
	var reqCancel context.CancelFunc
	if s.timeout > 0 {
		reqCtx, reqCancel = context.WithTimeout(s.ctx, s.timeout)
	} else {
		reqCtx, reqCancel = context.WithCancel(s.ctx)
	}

	var req *http.Request
	var err error

	switch s.apiFormat {
	case FormatOllama:
		req, err = s.buildOllamaRequest(reqCtx, messages)
	case FormatOpenAI:
		req, err = s.buildOpenAIRequest(reqCtx, messages)
	default:
		reqCancel()
		return fmt.Errorf("localmodel: unknown api_format %q", s.apiFormat)
	}
	if err != nil {
		reqCancel()
		return err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		reqCancel()
		return fmt.Errorf("localmodel: request failed: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		reqCancel()
		return fmt.Errorf("localmodel: API returned %d", resp.StatusCode)
	}

	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		defer reqCancel()
		defer resp.Body.Close()

		switch s.apiFormat {
		case FormatOllama:
			s.readOllamaStream(resp)
		case FormatOpenAI:
			s.readOpenAIStream(resp)
		}
	}()

	return nil
}

// ── Ollama native API (/api/chat) ──────────────────────────────

func (s *localSession) buildOllamaRequest(ctx context.Context, messages []chatMessage) (*http.Request, error) {
	body := struct {
		Model    string        `json:"model"`
		Messages []chatMessage `json:"messages"`
		Stream   bool          `json:"stream"`
	}{
		Model:    s.model,
		Messages: messages,
		Stream:   true,
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("localmodel: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/api/chat", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("localmodel: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

type ollamaChunk struct {
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done          bool   `json:"done"`
	TotalDuration int64  `json:"total_duration,omitempty"`
	Error         string `json:"error,omitempty"`
}

func (s *localSession) readOllamaStream(resp *http.Response) {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1*1024*1024)

	var fullContent strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var chunk ollamaChunk
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			slog.Debug("localmodel: non-JSON line", "line", line)
			continue
		}

		if chunk.Error != "" {
			s.emitError(fmt.Errorf("localmodel: %s", chunk.Error))
			return
		}

		if chunk.Message.Content != "" {
			fullContent.WriteString(chunk.Message.Content)
			s.emitText(chunk.Message.Content)
		}

		if chunk.Done {
			s.appendAssistant(fullContent.String())
			s.emitResult()
			return
		}
	}

	if err := scanner.Err(); err != nil {
		s.emitError(fmt.Errorf("localmodel: read stream: %w", err))
	}
}

// ── OpenAI-compatible API (/v1/chat/completions) ───────────────

func (s *localSession) buildOpenAIRequest(ctx context.Context, messages []chatMessage) (*http.Request, error) {
	body := struct {
		Model    string        `json:"model"`
		Messages []chatMessage `json:"messages"`
		Stream   bool          `json:"stream"`
	}{
		Model:    s.model,
		Messages: messages,
		Stream:   true,
	}

	bodyBytes, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("localmodel: marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("localmodel: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if s.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+s.apiKey)
	}
	return req, nil
}

type openAIChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (s *localSession) readOpenAIStream(resp *http.Response) {
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1*1024*1024)

	var fullContent strings.Builder

	for scanner.Scan() {
		line := scanner.Text()

		// SSE format: "data: {...}" or "data: [DONE]"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			s.appendAssistant(fullContent.String())
			s.emitResult()
			return
		}

		var chunk openAIChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			slog.Debug("localmodel: non-JSON SSE data", "data", data)
			continue
		}

		if chunk.Error != nil && chunk.Error.Message != "" {
			s.emitError(fmt.Errorf("localmodel: %s", chunk.Error.Message))
			return
		}

		if len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			if content != "" {
				fullContent.WriteString(content)
				s.emitText(content)
			}

			if chunk.Choices[0].FinishReason != nil {
				s.appendAssistant(fullContent.String())
				s.emitResult()
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		s.emitError(fmt.Errorf("localmodel: read stream: %w", err))
	}
}

// ── Shared helpers ─────────────────────────────────────────────

func (s *localSession) appendAssistant(content string) {
	s.historyMu.Lock()
	s.history = append(s.history, chatMessage{Role: "assistant", Content: content})
	s.historyMu.Unlock()
}

func (s *localSession) emitText(content string) {
	evt := core.Event{Type: core.EventText, Content: content}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
	}
}

func (s *localSession) emitResult() {
	evt := core.Event{Type: core.EventResult, SessionID: s.sessionID, Done: true}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
	}
}

func (s *localSession) emitError(err error) {
	evt := core.Event{Type: core.EventError, Error: err}
	select {
	case s.events <- evt:
	case <-s.ctx.Done():
	}
}

func (s *localSession) RespondPermission(_ string, _ core.PermissionResult) error {
	return nil
}

func (s *localSession) Events() <-chan core.Event {
	return s.events
}

func (s *localSession) CurrentSessionID() string {
	return s.sessionID
}

func (s *localSession) Alive() bool {
	return s.alive.Load()
}

func (s *localSession) Close() error {
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
		slog.Warn("localmodel: close timed out")
	}
	close(s.events)
	return nil
}
