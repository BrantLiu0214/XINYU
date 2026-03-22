import { useEffect, useRef, useState } from 'react';
import type { Message, UserSession } from '../types';
import { streamChat } from '../hooks/useStreamChat';
import { CrisisCard } from '../components/CrisisCard';
import { RiskBadge } from '../components/RiskBadge';
import styles from './ChatPage.module.css';

const VISITOR_KEY = 'xinyu_visitor_id';

// ── Session chooser overlay ───────────────────────────────────────────────

interface ChooserProps {
  sessions: UserSession[];
  onSelect: (sessionId: string) => void;
  onNew: () => void;
}

function SessionChooser({ sessions, onSelect, onNew }: ChooserProps) {
  return (
    <div className={styles.chooserOverlay}>
      <div className={styles.chooserCard}>
        <div className={styles.chooserTitle}>欢迎回来</div>
        <p className={styles.chooserSub}>选择一个过去的对话继续，或开始新的对话</p>
        <div className={styles.chooserList}>
          {sessions.map(s => (
            <button
              key={s.session_id}
              className={styles.chooserItem}
              onClick={() => onSelect(s.session_id)}
            >
              <span className={styles.chooserDate}>
                {new Date(s.started_at).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })}
              </span>
              <span className={styles.chooserCount}>{s.message_count} 条消息</span>
              <RiskBadge level={s.latest_risk_level} small />
              <span className={styles.chooserArrow}>›</span>
            </button>
          ))}
        </div>
        <button className={styles.newSessionBtn} onClick={onNew}>
          ＋ 开始新对话
        </button>
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────

export function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [pastSessions, setPastSessions] = useState<UserSession[] | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const visitorId = localStorage.getItem(VISITOR_KEY);
    if (visitorId) {
      // Returning visitor — fetch their sessions.
      fetch(`/api/v1/sessions?visitor_id=${encodeURIComponent(visitorId)}`)
        .then(r => r.json())
        .then((data: { sessions: UserSession[] }) => {
          if (data.sessions && data.sessions.length > 0) {
            setPastSessions(data.sessions);
          } else {
            // No sessions yet (edge case) — create a new one.
            createNewSession(visitorId);
          }
        })
        .catch(() => createNewSession(visitorId));
    } else {
      // New visitor — create session immediately.
      createNewSession(null);
    }
  }, []);

  // Auto-scroll to bottom on new content.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  function createNewSession(visitorId: string | null) {
    const body = visitorId
      ? JSON.stringify({ visitor_id: visitorId })
      : '{}';
    fetch('/api/v1/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body })
      .then(r => r.json())
      .then(d => {
        localStorage.setItem(VISITOR_KEY, d.visitor_id);
        setSessionId(d.session_id);
        setPastSessions(null);
      })
      .catch(console.error);
  }

  function loadExistingSession(sid: string) {
    fetch(`/api/v1/dashboard/sessions/${sid}/messages`)
      .then(r => r.json())
      .then(data => {
        const loaded: Message[] = (data.messages ?? []).map((m: { message_id: string; role: string; content: string }) => ({
          id: m.message_id,
          role: m.role as 'user' | 'assistant',
          content: m.content,
        }));
        setMessages(loaded);
        setSessionId(sid);
        setPastSessions(null);
      })
      .catch(console.error);
  }

  async function handleSend() {
    if (!sessionId || !input.trim() || sending) return;
    const text = input.trim();
    setInput('');
    setSending(true);

    const userMsgId = crypto.randomUUID();
    const assistantMsgId = crypto.randomUUID();

    setMessages(prev => [...prev, { id: userMsgId, role: 'user', content: text }]);
    setMessages(prev => [...prev, { id: assistantMsgId, role: 'assistant', content: '', streaming: true }]);

    await streamChat(sessionId, text, {
      onMeta: (meta) => {
        setMessages(prev => prev.map(m => m.id === userMsgId ? { ...m, meta } : m));
      },
      onAlert: (alert) => {
        setMessages(prev => prev.map(m => m.id === assistantMsgId ? { ...m, alert } : m));
      },
      onToken: (token) => {
        setMessages(prev => prev.map(m =>
          m.id === assistantMsgId ? { ...m, content: m.content + token } : m
        ));
      },
      onComplete: () => {
        setMessages(prev => prev.map(m =>
          m.id === assistantMsgId ? { ...m, streaming: false } : m
        ));
        setSending(false);
      },
      onError: (err) => {
        console.error(err);
        setMessages(prev => prev.map(m =>
          m.id === assistantMsgId
            ? { ...m, content: '（发送失败，请稍后重试）', streaming: false }
            : m
        ));
        setSending(false);
      },
    });
  }

  // Show session chooser for returning visitors.
  if (pastSessions !== null) {
    return (
      <SessionChooser
        sessions={pastSessions}
        onSelect={loadExistingSession}
        onNew={() => createNewSession(localStorage.getItem(VISITOR_KEY))}
      />
    );
  }

  if (!sessionId) {
    return <div className={styles.loading}>正在初始化会话…</div>;
  }

  return (
    <div className={styles.page}>
      <div className={styles.messages}>
        {messages.length === 0 && (
          <p className={styles.placeholder}>你好，我是心语。有什么想和我说的吗？</p>
        )}
        {messages.map(msg => (
          <div key={msg.id} className={`${styles.row} ${msg.role === 'user' ? styles.rowUser : styles.rowAssistant}`}>
            <div className={`${styles.bubble} ${msg.role === 'user' ? styles.bubbleUser : styles.bubbleAssistant}`}>
              {msg.role === 'assistant' && msg.alert && <CrisisCard alert={msg.alert} />}
              <span>
                {msg.content}
                {msg.streaming && <span className={styles.cursor}>▋</span>}
              </span>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form
        className={styles.inputArea}
        onSubmit={e => { e.preventDefault(); handleSend(); }}
      >
        <input
          className={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder="输入消息，按 Enter 发送…"
          disabled={sending}
          autoFocus
        />
        <button className={styles.sendBtn} type="submit" disabled={sending || !input.trim()}>
          发送
        </button>
      </form>
    </div>
  );
}

