import { useState } from 'react';
import { getRole, saveAuth } from '../auth';
import styles from './RequireAuth.module.css';

export const COUNSELOR_STORAGE_KEY = 'xinyu_role'; // kept for NavBar import compat

interface Props {
  children: React.ReactNode;
}

export function RequireAuth({ children }: Props) {
  const [authed, setAuthed] = useState(() => getRole() === 'counselor');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  if (authed) return <>{children}</>;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const res = await fetch('/api/v1/auth/counselor/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      if (!res.ok) {
        setError('用户名或密码错误');
        setPassword('');
        return;
      }
      const d = await res.json();
      saveAuth(d.access_token, d.role);
      setAuthed(true);
    } catch {
      setError('网络错误，请重试');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.card}>
        <div className={styles.title}>咨询师后台</div>
        <p className={styles.subtitle}>请输入咨询师账户</p>
        <form onSubmit={handleSubmit}>
          <input
            className={styles.input}
            type="text"
            value={username}
            onChange={e => { setUsername(e.target.value); setError(''); }}
            placeholder="用户名"
            autoFocus
            required
          />
          <input
            className={styles.input}
            type="password"
            value={password}
            onChange={e => { setPassword(e.target.value); setError(''); }}
            placeholder="密码"
            required
          />
          {error && <p className={styles.error}>{error}</p>}
          <button className={styles.btn} type="submit" disabled={loading || !username || !password}>
            {loading ? '请稍候…' : '登录'}
          </button>
        </form>
      </div>
    </div>
  );
}
