package slack

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/chenhg5/cc-connect/core"

	"github.com/slack-go/slack"
	"github.com/slack-go/slack/slackevents"
	"github.com/slack-go/slack/socketmode"
)

func init() {
	core.RegisterPlatform("slack", New)
}

type replyContext struct {
	channel   string
	timestamp string // thread_ts for threading replies
}

// ---------------------------------------------------------------------------
// Shared socket pool: one WebSocket per (bot_token, app_token) pair.
//
// When multiple projects use the same Slack bot token, Slack delivers each
// event to only ONE of the WebSocket connections.  If every project opens
// its own connection, events are randomly distributed and channel-based
// routing breaks.
//
// The pool ensures a single connection is shared.  Incoming events are
// dispatched to every registered handler; each Platform instance filters
// by channel_ids and ignores events that don't belong to it.
// ---------------------------------------------------------------------------

type sharedSocket struct {
	client   *slack.Client
	socket   *socketmode.Client
	cancel   context.CancelFunc
	handlers []eventHandler
	mu       sync.Mutex
}

type eventHandler struct {
	platform *Platform
	handler  core.MessageHandler
}

var (
	socketPool   = make(map[string]*sharedSocket) // key: bot_token
	socketPoolMu sync.Mutex
)

func getOrCreateSocket(botToken, appToken string) *sharedSocket {
	socketPoolMu.Lock()
	defer socketPoolMu.Unlock()

	if ss, ok := socketPool[botToken]; ok {
		return ss
	}

	client := slack.New(botToken, slack.OptionAppLevelToken(appToken))
	sock := socketmode.New(client)

	ctx, cancel := context.WithCancel(context.Background())
	ss := &sharedSocket{
		client: client,
		socket: sock,
		cancel: cancel,
	}

	// Single event loop dispatches to all registered handlers
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case evt := <-sock.Events:
				ss.dispatch(evt)
			}
		}
	}()

	go func() {
		if err := sock.RunContext(ctx); err != nil {
			slog.Error("slack: socket mode error", "error", err)
		}
	}()

	socketPool[botToken] = ss
	slog.Info("slack: socket mode connected (shared)")
	return ss
}

func (ss *sharedSocket) addHandler(p *Platform, h core.MessageHandler) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	ss.handlers = append(ss.handlers, eventHandler{platform: p, handler: h})
}

func (ss *sharedSocket) dispatch(evt socketmode.Event) {
	slog.Debug("slack: raw event received", "type", evt.Type)
	switch evt.Type {
	case socketmode.EventTypeEventsAPI:
		data, ok := evt.Data.(slackevents.EventsAPIEvent)
		if !ok {
			return
		}
		if evt.Request != nil {
			ss.socket.Ack(*evt.Request)
		}

		// Fan out to all registered handlers — each one filters by channel_ids
		ss.mu.Lock()
		handlers := make([]eventHandler, len(ss.handlers))
		copy(handlers, ss.handlers)
		ss.mu.Unlock()

		for _, eh := range handlers {
			eh.platform.handleEventsAPI(data)
		}

	case socketmode.EventTypeConnecting:
		slog.Debug("slack: connecting...")
	case socketmode.EventTypeConnected:
		slog.Info("slack: connected")
	case socketmode.EventTypeConnectionError:
		slog.Error("slack: connection error")
	}
}

func (ss *sharedSocket) stop() {
	ss.cancel()
}

// ---------------------------------------------------------------------------
// Platform — one per project, filters events by channel_ids
// ---------------------------------------------------------------------------

type Platform struct {
	botToken              string
	appToken              string
	allowFrom             string
	channelIDs            map[string]bool
	shareSessionInChannel bool
	client                *slack.Client
	handler               core.MessageHandler
	channelNameCache      map[string]string
	channelCacheMu        sync.RWMutex
	userNameCache         sync.Map
	shared                *sharedSocket
}

func New(opts map[string]any) (core.Platform, error) {
	botToken, _ := opts["bot_token"].(string)
	appToken, _ := opts["app_token"].(string)
	allowFrom, _ := opts["allow_from"].(string)
	core.CheckAllowFrom("slack", allowFrom)
	shareSessionInChannel, _ := opts["share_session_in_channel"].(bool)
	if botToken == "" || appToken == "" {
		return nil, fmt.Errorf("slack: bot_token and app_token are required")
	}

	channelIDs := make(map[string]bool)
	if raw, _ := opts["channel_ids"].(string); raw != "" {
		for _, id := range strings.Split(raw, ",") {
			id = strings.TrimSpace(id)
			if id != "" {
				channelIDs[id] = true
			}
		}
	}

	return &Platform{
		botToken:              botToken,
		appToken:              appToken,
		allowFrom:             allowFrom,
		channelIDs:            channelIDs,
		shareSessionInChannel: shareSessionInChannel,
		channelNameCache:      make(map[string]string),
	}, nil
}

func (p *Platform) Name() string { return "slack" }

func (p *Platform) Start(handler core.MessageHandler) error {
	p.handler = handler

	// Get or create a shared socket for this bot token
	ss := getOrCreateSocket(p.botToken, p.appToken)
	p.shared = ss
	p.client = ss.client

	// Register this platform as a handler on the shared socket
	ss.addHandler(p, handler)

	slog.Info("slack: platform registered on shared socket", "channels", len(p.channelIDs))
	return nil
}

// handleEventsAPI processes an EventsAPI event, filtering by channel_ids.
func (p *Platform) handleEventsAPI(data slackevents.EventsAPIEvent) {
	if data.Type != slackevents.CallbackEvent {
		return
	}

	switch ev := data.InnerEvent.Data.(type) {
	case *slackevents.AppMentionEvent:
		p.handleAppMention(ev)
	case *slackevents.MessageEvent:
		p.handleMessage(ev)
	}
}

func (p *Platform) handleAppMention(ev *slackevents.AppMentionEvent) {
	if ev.BotID != "" || ev.User == "" {
		return
	}

	if ts := ev.TimeStamp; ts != "" {
		if dotIdx := strings.IndexByte(ts, '.'); dotIdx > 0 {
			if sec, err := strconv.ParseInt(ts[:dotIdx], 10, 64); err == nil {
				if core.IsOldMessage(time.Unix(sec, 0)) {
					return
				}
			}
		}
	}

	// Channel filter
	if len(p.channelIDs) > 0 && !p.channelIDs[ev.Channel] {
		return
	}

	slog.Debug("slack: app_mention received", "user", ev.User, "channel", ev.Channel)

	if !core.AllowList(p.allowFrom, ev.User) {
		slog.Debug("slack: app_mention from unauthorized user", "user", ev.User)
		return
	}

	var sessionKey string
	if p.shareSessionInChannel {
		sessionKey = fmt.Sprintf("slack:%s", ev.Channel)
	} else {
		sessionKey = fmt.Sprintf("slack:%s:%s", ev.Channel, ev.User)
	}

	msg := &core.Message{
		SessionKey: sessionKey, Platform: "slack",
		UserID: ev.User, UserName: p.resolveUserName(ev.User),
		ChatName: p.resolveChannelNameForMsg(ev.Channel),
		Content:   stripAppMentionText(ev.Text),
		MessageID: ev.TimeStamp,
		ReplyCtx:  replyContext{channel: ev.Channel, timestamp: ev.TimeStamp},
	}
	if msg.Content == "" {
		return
	}
	p.handler(p, msg)
}

func (p *Platform) handleMessage(ev *slackevents.MessageEvent) {
	if ev.BotID != "" || ev.User == "" {
		return
	}

	if ts := ev.TimeStamp; ts != "" {
		if dotIdx := strings.IndexByte(ts, '.'); dotIdx > 0 {
			if sec, err := strconv.ParseInt(ts[:dotIdx], 10, 64); err == nil {
				if core.IsOldMessage(time.Unix(sec, 0)) {
					return
				}
			}
		}
	}

	// Channel filter
	if len(p.channelIDs) > 0 && !p.channelIDs[ev.Channel] {
		return
	}

	slog.Debug("slack: message received", "user", ev.User, "channel", ev.Channel)

	if !core.AllowList(p.allowFrom, ev.User) {
		slog.Debug("slack: message from unauthorized user", "user", ev.User)
		return
	}

	var sessionKey string
	if p.shareSessionInChannel {
		sessionKey = fmt.Sprintf("slack:%s", ev.Channel)
	} else {
		sessionKey = fmt.Sprintf("slack:%s:%s", ev.Channel, ev.User)
	}
	ts := ev.TimeStamp

	var images []core.ImageAttachment
	var audio *core.AudioAttachment
	for _, f := range ev.Files {
		if f.Mimetype != "" && strings.HasPrefix(f.Mimetype, "audio/") {
			data, err := p.downloadSlackFile(f.URLPrivateDownload)
			if err != nil {
				slog.Error("slack: download audio failed", "error", err)
				continue
			}
			format := "mp3"
			if parts := strings.SplitN(f.Mimetype, "/", 2); len(parts) == 2 {
				format = parts[1]
			}
			audio = &core.AudioAttachment{
				MimeType: f.Mimetype, Data: data, Format: format,
			}
		} else if f.Mimetype != "" && strings.HasPrefix(f.Mimetype, "image/") {
			imgData, err := p.downloadSlackFile(f.URLPrivateDownload)
			if err != nil {
				slog.Error("slack: download file failed", "error", err)
				continue
			}
			images = append(images, core.ImageAttachment{
				MimeType: f.Mimetype, Data: imgData, FileName: f.Name,
			})
		}
	}

	if ev.Text == "" && len(images) == 0 && audio == nil {
		return
	}

	msg := &core.Message{
		SessionKey: sessionKey, Platform: "slack",
		UserID: ev.User, UserName: p.resolveUserName(ev.User),
		ChatName: p.resolveChannelNameForMsg(ev.Channel),
		Content: ev.Text, Images: images, Audio: audio,
		MessageID: ts,
		ReplyCtx: replyContext{channel: ev.Channel, timestamp: ts},
	}
	p.handler(p, msg)
}

func stripAppMentionText(text string) string {
	if idx := strings.Index(text, "> "); idx != -1 && strings.HasPrefix(text, "<@") {
		return strings.TrimSpace(text[idx+2:])
	}
	return text
}

func (p *Platform) Reply(ctx context.Context, rctx any, content string) error {
	rc, ok := rctx.(replyContext)
	if !ok {
		return fmt.Errorf("slack: invalid reply context type %T", rctx)
	}

	opts := []slack.MsgOption{
		slack.MsgOptionText(content, false),
	}
	if rc.timestamp != "" {
		opts = append(opts, slack.MsgOptionTS(rc.timestamp))
	}

	_, _, err := p.client.PostMessageContext(ctx, rc.channel, opts...)
	if err != nil {
		return fmt.Errorf("slack: send: %w", err)
	}
	return nil
}

// Send sends a new message (not a reply)
func (p *Platform) Send(ctx context.Context, rctx any, content string) error {
	rc, ok := rctx.(replyContext)
	if !ok {
		return fmt.Errorf("slack: invalid reply context type %T", rctx)
	}

	_, _, err := p.client.PostMessageContext(ctx, rc.channel, slack.MsgOptionText(content, false))
	if err != nil {
		return fmt.Errorf("slack: send: %w", err)
	}
	return nil
}

func (p *Platform) downloadSlackFile(url string) ([]byte, error) {
	if url == "" {
		return nil, fmt.Errorf("empty URL")
	}
	req, _ := http.NewRequest("GET", url, nil)
	req.Header.Set("Authorization", "Bearer "+p.botToken)
	resp, err := core.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s", core.RedactToken(err.Error(), p.botToken))
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func (p *Platform) ReconstructReplyCtx(sessionKey string) (any, error) {
	// slack:{channel}:{user}
	parts := strings.SplitN(sessionKey, ":", 3)
	if len(parts) < 2 || parts[0] != "slack" {
		return nil, fmt.Errorf("slack: invalid session key %q", sessionKey)
	}
	return replyContext{channel: parts[1]}, nil
}

func (p *Platform) resolveUserName(userID string) string {
	if cached, ok := p.userNameCache.Load(userID); ok {
		return cached.(string)
	}
	user, err := p.client.GetUserInfo(userID)
	if err != nil {
		slog.Debug("slack: resolve user name failed", "user", userID, "error", err)
		return userID
	}
	name := user.RealName
	if name == "" {
		name = user.Profile.DisplayName
	}
	if name == "" {
		name = userID
	}
	p.userNameCache.Store(userID, name)
	return name
}

func (p *Platform) resolveChannelNameForMsg(channelID string) string {
	name, err := p.ResolveChannelName(channelID)
	if err != nil || name == "" {
		return channelID
	}
	return name
}

func (p *Platform) ResolveChannelName(channelID string) (string, error) {
	p.channelCacheMu.RLock()
	if name, ok := p.channelNameCache[channelID]; ok {
		p.channelCacheMu.RUnlock()
		return name, nil
	}
	p.channelCacheMu.RUnlock()

	info, err := p.client.GetConversationInfo(&slack.GetConversationInfoInput{
		ChannelID: channelID,
	})
	if err != nil {
		return "", fmt.Errorf("slack: resolve channel name for %s: %w", channelID, err)
	}

	p.channelCacheMu.Lock()
	p.channelNameCache[channelID] = info.Name
	p.channelCacheMu.Unlock()

	return info.Name, nil
}

func (p *Platform) Stop() error {
	// Don't stop the shared socket — other projects may still use it.
	// The socket is cleaned up when the process exits.
	return nil
}
