package localmodel

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/chenhg5/cc-connect/core"
)

func init() {
	core.RegisterAgent("localmodel", New)
}

// APIFormat determines which HTTP API protocol to use.
type APIFormat string

const (
	// FormatOllama uses Ollama's native /api/chat endpoint.
	FormatOllama APIFormat = "ollama"
	// FormatOpenAI uses the OpenAI-compatible /v1/chat/completions endpoint.
	// Supported by: Ollama, vLLM, llama.cpp, LM Studio, LocalAI, text-generation-webui, etc.
	FormatOpenAI APIFormat = "openai"
)

// Agent drives a local LLM server via its HTTP API.
//
// Supports two API formats:
//   - "ollama":  Ollama native API (/api/chat with NDJSON streaming)
//   - "openai":  OpenAI-compatible API (/v1/chat/completions with SSE streaming)
//
// Auto-detection: if api_format is not set, the agent probes /api/tags (Ollama)
// and falls back to OpenAI-compatible mode.
//
// This makes the agent compatible with Ollama, vLLM, llama.cpp server,
// LM Studio, LocalAI, and any other server exposing an OpenAI-compatible API.
type Agent struct {
	baseURL   string
	model     string
	workDir   string
	apiFormat APIFormat
	apiKey    string // optional, for OpenAI-compatible servers that require auth
	timeout   time.Duration
	mu        sync.Mutex
}

func New(opts map[string]any) (core.Agent, error) {
	baseURL, _ := opts["base_url"].(string)
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}
	baseURL = strings.TrimRight(baseURL, "/")

	model, _ := opts["model"].(string)
	if model == "" {
		model = "qwen3:8b"
	}

	workDir, _ := opts["work_dir"].(string)
	if workDir == "" {
		workDir = "."
	}

	apiKey, _ := opts["api_key"].(string)

	var timeoutMins int64
	switch v := opts["timeout_mins"].(type) {
	case int64:
		timeoutMins = v
	case int:
		timeoutMins = int64(v)
	case float64:
		timeoutMins = int64(v)
	}
	var timeout time.Duration
	if timeoutMins > 0 {
		timeout = time.Duration(timeoutMins) * time.Minute
	} else {
		timeout = 5 * time.Minute
	}

	// Determine API format
	formatStr, _ := opts["api_format"].(string)
	apiFormat := resolveAPIFormat(formatStr, baseURL)

	slog.Info("localmodel: initialized", "base_url", baseURL, "model", model, "api_format", apiFormat)

	return &Agent{
		baseURL:   baseURL,
		model:     model,
		workDir:   workDir,
		apiFormat: apiFormat,
		apiKey:    apiKey,
		timeout:   timeout,
	}, nil
}

// resolveAPIFormat determines which API format to use.
// If explicit, use it. Otherwise, probe the server.
func resolveAPIFormat(explicit, baseURL string) APIFormat {
	switch strings.ToLower(strings.TrimSpace(explicit)) {
	case "ollama":
		return FormatOllama
	case "openai":
		return FormatOpenAI
	}

	// Auto-detect: try Ollama's /api/tags endpoint
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"/api/tags", nil)
	if err != nil {
		return FormatOpenAI
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		// Can't reach /api/tags → assume OpenAI-compatible
		return FormatOpenAI
	}
	resp.Body.Close()
	if resp.StatusCode == http.StatusOK {
		return FormatOllama
	}
	return FormatOpenAI
}

func (a *Agent) Name() string { return "localmodel" }

func (a *Agent) StartSession(ctx context.Context, sessionID string) (core.AgentSession, error) {
	a.mu.Lock()
	model := a.model
	baseURL := a.baseURL
	timeout := a.timeout
	apiFormat := a.apiFormat
	apiKey := a.apiKey
	a.mu.Unlock()

	return newSession(ctx, baseURL, model, sessionID, timeout, apiFormat, apiKey)
}

func (a *Agent) ListSessions(_ context.Context) ([]core.AgentSessionInfo, error) {
	return nil, nil
}

func (a *Agent) Stop() error { return nil }

// ── ModelSwitcher ────────────────────────────────────────────────

func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
	slog.Info("localmodel: model changed", "model", model)
}

func (a *Agent) GetModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.model
}

func (a *Agent) AvailableModels(ctx context.Context) []core.ModelOption {
	a.mu.Lock()
	baseURL := a.baseURL
	apiFormat := a.apiFormat
	apiKey := a.apiKey
	a.mu.Unlock()

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	switch apiFormat {
	case FormatOllama:
		return a.fetchOllamaModels(reqCtx, baseURL)
	case FormatOpenAI:
		return a.fetchOpenAIModels(reqCtx, baseURL, apiKey)
	}
	return nil
}

func (a *Agent) fetchOllamaModels(ctx context.Context, baseURL string) []core.ModelOption {
	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"/api/tags", nil)
	if err != nil {
		return nil
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	var result struct {
		Models []struct {
			Name string `json:"name"`
			Size int64  `json:"size"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil
	}

	var models []core.ModelOption
	for _, m := range result.Models {
		sizeGB := float64(m.Size) / (1024 * 1024 * 1024)
		models = append(models, core.ModelOption{Name: m.Name, Desc: fmt.Sprintf("%.1f GB", sizeGB)})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models
}

func (a *Agent) fetchOpenAIModels(ctx context.Context, baseURL, apiKey string) []core.ModelOption {
	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"/v1/models", nil)
	if err != nil {
		return nil
	}
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	var result struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil
	}

	var models []core.ModelOption
	for _, m := range result.Data {
		models = append(models, core.ModelOption{Name: m.ID})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models
}

// ── WorkDirSwitcher ────────────────────────────────────────────

func (a *Agent) SetWorkDir(dir string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.workDir = dir
	slog.Info("localmodel: work_dir changed", "work_dir", dir)
}

func (a *Agent) GetWorkDir() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.workDir
}
