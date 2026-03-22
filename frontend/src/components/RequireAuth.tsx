import { useState } from 'react';
import styles from './RequireAuth.module.css';

const COUNSELOR_PASSCODE = 'xinyu2026';
export const COUNSELOR_STORAGE_KEY = 'xinyu_counselor';

interface Props {
  children: React.ReactNode;
}

export function RequireAuth({ children }: Props) {
  const [authed, setAuthed] = useState(
    () => sessionStorage.getItem(COUNSELOR_STORAGE_KEY) === '1'
  );
  const [input, setInput] = useState('');
  const [error, setError] = useState('');

  if (authed) return <>{children}</>;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (input === COUNSELOR_PASSCODE) {
      sessionStorage.setItem(COUNSELOR_STORAGE_KEY, '1');
      setAuthed(true);
    } else {
      setError('密码错误，请重试');
      setInput('');
    }
  }

  return (
    <div className={styles.wrapper}>
      <div className={styles.card}>
        <div className={styles.title}>咨询师后台</div>
        <p className={styles.subtitle}>请输入访问密码</p>
        <form onSubmit={handleSubmit}>
          <input
            className={styles.input}
            type="password"
            value={input}
            onChange={e => { setInput(e.target.value); setError(''); }}
            placeholder="密码"
            autoFocus
          />
          {error && <p className={styles.error}>{error}</p>}
          <button className={styles.btn} type="submit" disabled={!input}>
            进入
          </button>
        </form>
      </div>
    </div>
  );
}
