package ollama

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
	core.RegisterAgent("ollama", New)
}

// Agent drives a local Ollama instance via its HTTP API.
//
// Unlike other cc-connect agents that wrap CLI tools, this agent communicates
// with the Ollama REST API directly (/api/chat with streaming). This makes it
// suitable for local LLMs (Qwen, Llama, Mistral, etc.) running on the same
// machine or network.
type Agent struct {
	baseURL string
	model   string
	workDir string
	timeout time.Duration
	mu      sync.Mutex
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

	// Verify Ollama is reachable
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", baseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama: cannot reach %s — is Ollama running? %w", baseURL, err)
	}
	resp.Body.Close()

	return &Agent{
		baseURL: baseURL,
		model:   model,
		workDir: workDir,
		timeout: timeout,
	}, nil
}

func (a *Agent) Name() string { return "ollama" }

func (a *Agent) StartSession(ctx context.Context, sessionID string) (core.AgentSession, error) {
	a.mu.Lock()
	model := a.model
	baseURL := a.baseURL
	timeout := a.timeout
	a.mu.Unlock()

	return newSession(ctx, baseURL, model, sessionID, timeout)
}

func (a *Agent) ListSessions(_ context.Context) ([]core.AgentSessionInfo, error) {
	// Ollama API is stateless; sessions are maintained in-memory by this agent.
	return nil, nil
}

func (a *Agent) Stop() error { return nil }

// ── ModelSwitcher ────────────────────────────────────────────────

func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
	slog.Info("ollama: model changed", "model", model)
}

func (a *Agent) GetModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.model
}

func (a *Agent) AvailableModels(ctx context.Context) []core.ModelOption {
	a.mu.Lock()
	baseURL := a.baseURL
	a.mu.Unlock()

	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, "GET", baseURL+"/api/tags", nil)
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
			Name       string `json:"name"`
			ModifiedAt string `json:"modified_at"`
			Size       int64  `json:"size"`
		} `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil
	}

	var models []core.ModelOption
	for _, m := range result.Models {
		sizeGB := float64(m.Size) / (1024 * 1024 * 1024)
		desc := fmt.Sprintf("%.1f GB", sizeGB)
		models = append(models, core.ModelOption{Name: m.Name, Desc: desc})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models
}

// ── WorkDirSwitcher ────────────────────────────────────────────

func (a *Agent) SetWorkDir(dir string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.workDir = dir
	slog.Info("ollama: work_dir changed", "work_dir", dir)
}

func (a *Agent) GetWorkDir() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.workDir
}
