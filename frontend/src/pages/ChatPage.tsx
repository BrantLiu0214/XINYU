import { useEffect, useRef, useState } from 'react';
import type { Message, UserSession } from '../types';
import { streamChat } from '../hooks/useStreamChat';
import { CrisisCard } from '../components/CrisisCard';
import { RiskBadge } from '../components/RiskBadge';
import { getToken, getRole, saveAuth, clearAuth, authHeaders } from '../auth';
import { COLLEGES } from '../constants';
import styles from './ChatPage.module.css';

// ── Auth form ─────────────────────────────────────────────────────────────

function AuthForm() {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [realName, setRealName] = useState('');
  const [college, setCollege] = useState('');
  const [studentId, setStudentId] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    const url = mode === 'login'
      ? '/api/v1/auth/visitor/login'
      : '/api/v1/auth/visitor/register';
    const body = mode === 'login'
      ? { username, password }
      : {
          username, password,
          display_name: displayName || undefined,
          real_name: realName || undefined,
          college: college || undefined,
          student_id: studentId || undefined,
        };
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setError(d.detail ?? (mode === 'login' ? '用户名或密码错误' : '注册失败，用户名可能已被使用'));
        return;
      }
      const d = await res.json();
      saveAuth(d.access_token, d.role);
      window.location.href = '/';
    } catch {
      setError('网络错误，请重试');
    } finally {
      setLoading(false);
    }
  }

  async function handleGuest() {
    setError('');
    setLoading(true);
    try {
      const res = await fetch('/api/v1/auth/visitor/guest', { method: 'POST' });
      if (!res.ok) { setError('游客登录失败，请重试'); return; }
      const d = await res.json();
      saveAuth(d.access_token, d.role);
      window.location.href = '/';
    } catch {
      setError('网络错误，请重试');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.chooserOverlay}>
      <div className={styles.chooserCard}>
        <div className={styles.chooserTitle}>欢迎使用心语</div>
        <div className={styles.authTabs}>
          <button
            className={mode === 'login' ? styles.authTabActive : styles.authTab}
            onClick={() => { setMode('login'); setError(''); }}
          >登录</button>
          <button
            className={mode === 'register' ? styles.authTabActive : styles.authTab}
            onClick={() => { setMode('register'); setError(''); }}
          >注册</button>
        </div>
        <form onSubmit={handleSubmit} className={styles.authForm}>
          <input
            className={styles.authInput}
            type="text"
            placeholder="用户名"
            value={username}
            onChange={e => setUsername(e.target.value)}
            autoFocus
            required
          />
          {mode === 'register' && (
            <>
              <input
                className={styles.authInput}
                type="text"
                placeholder="昵称（可选）"
                value={displayName}
                onChange={e => setDisplayName(e.target.value)}
              />
              <input
                className={styles.authInput}
                type="text"
                placeholder="真实姓名"
                value={realName}
                onChange={e => setRealName(e.target.value)}
                required
              />
              <select
                className={styles.authSelect}
                value={college}
                onChange={e => setCollege(e.target.value)}
                required
              >
                <option value="" disabled>请选择学院</option>
                {COLLEGES.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
              <input
                className={styles.authInput}
                type="text"
                placeholder="学号"
                value={studentId}
                onChange={e => setStudentId(e.target.value)}
                required
              />
            </>
          )}
          <input
            className={styles.authInput}
            type="password"
            placeholder="密码"
            value={password}
            onChange={e => setPassword(e.target.value)}
            required
          />
          {error && <p className={styles.authError}>{error}</p>}
          <button
            className={styles.newSessionBtn}
            type="submit"
            disabled={loading || !username || !password || (mode === 'register' && (!realName || !college || !studentId))}
          >
            {loading ? '请稍候…' : mode === 'login' ? '登录' : '注册'}
          </button>
        </form>
        <div className={styles.authDivider}>或</div>
        <button
          className={styles.guestBtn}
          onClick={handleGuest}
          disabled={loading}
        >
          以游客身份聊天
        </button>
      </div>
    </div>
  );
}

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
  const [authenticated, setAuthenticated] = useState(
    () => getToken() !== null && getRole() === 'visitor'
  );
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!authenticated) return;
    fetch('/api/v1/sessions', { headers: authHeaders() })
      .then(r => {
        if (r.status === 401) { clearAuth(); setAuthenticated(false); return null; }
        return r.json();
      })
      .then((data: { sessions: UserSession[] } | null) => {
        if (!data) return;
        if (data.sessions && data.sessions.length > 0) {
          setPastSessions(data.sessions);
        } else {
          createNewSession();
        }
      })
      .catch(() => createNewSession());
  }, [authenticated]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  function createNewSession() {
    fetch('/api/v1/sessions', { method: 'POST', headers: authHeaders() })
      .then(r => r.json())
      .then(d => {
        setSessionId(d.session_id);
        setPastSessions(null);
      })
      .catch(console.error);
  }

  function loadExistingSession(sid: string) {
    fetch(`/api/v1/sessions/${sid}/messages`, { headers: authHeaders() })
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

  if (!authenticated) {
    return <AuthForm />;
  }

  if (pastSessions !== null) {
    return (
      <SessionChooser
        sessions={pastSessions}
        onSelect={loadExistingSession}
        onNew={createNewSession}
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
