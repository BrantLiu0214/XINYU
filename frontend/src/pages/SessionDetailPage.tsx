import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import type { MessageWithAnalysis } from '../types';
import type { ChartPoint } from '../components/EmotionChart';
import { EmotionChart } from '../components/EmotionChart';
import styles from './SessionDetailPage.module.css';

export function SessionDetailPage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const [messages, setMessages] = useState<MessageWithAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    fetch(`/api/v1/dashboard/sessions/${sessionId}/messages`)
      .then(r => {
        if (r.status === 404) { setNotFound(true); return null; }
        return r.json();
      })
      .then(d => {
        if (d) setMessages(d.messages ?? []);
        setLoading(false);
      })
      .catch(err => { console.error(err); setLoading(false); });
  }, [sessionId]);

  const chartData: ChartPoint[] = messages
    .filter(m => m.role === 'user' && m.analysis)
    .map((m, i) => ({
      turn: i + 1,
      intensity: m.analysis!.intensity_score,
      emotion: m.analysis!.emotion_label,
    }));

  if (loading) return <div className={styles.center}>加载中…</div>;
  if (notFound) return (
    <div className={styles.center}>
      <p>会话不存在</p>
      <Link to="/dashboard">← 返回后台</Link>
    </div>
  );

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <Link to="/dashboard" className={styles.back}>← 咨询师后台</Link>
        <span className={styles.sessionId}>会话 {sessionId?.slice(0, 8)}…</span>
      </div>

      <EmotionChart data={chartData} />

      <div className={styles.messages}>
        {messages.map(msg => (
          <div
            key={msg.message_id}
            className={`${styles.row} ${msg.role === 'user' ? styles.rowUser : styles.rowAssistant}`}
          >
            <div className={`${styles.bubble} ${msg.role === 'user' ? styles.bubbleUser : styles.bubbleAssistant}`}>
              <p className={styles.content}>{msg.content}</p>
              {msg.role === 'user' && msg.analysis && (
                <div className={styles.meta}>
                  {msg.analysis.emotion_label} · {msg.analysis.intent_label} · 强度 {(msg.analysis.intensity_score * 100).toFixed(0)}%
                </div>
              )}
            </div>
          </div>
        ))}
        {messages.length === 0 && <p className={styles.empty}>该会话暂无消息</p>}
      </div>
    </div>
  );
}
