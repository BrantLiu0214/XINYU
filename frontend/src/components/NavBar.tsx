import { Link, useLocation } from 'react-router-dom';
import { getRole, clearAuth } from '../auth';
import styles from './NavBar.module.css';

export function NavBar() {
  const { pathname } = useLocation();
  const role = getRole();

  function handleLogout() {
    clearAuth();
    window.location.href = '/';
  }

  return (
    <nav className={styles.nav}>
      <span className={styles.brand}>心语 XinYu</span>
      <div className={styles.links}>
        <Link className={pathname === '/' ? styles.active : ''} to="/">对话</Link>
        {role === 'counselor' && (
          <Link
            className={pathname.startsWith('/dashboard') || pathname.startsWith('/session') ? styles.active : ''}
            to="/dashboard"
          >
            咨询师后台
          </Link>
        )}
        {role && (
          <button className={styles.logoutBtn} onClick={handleLogout}>退出</button>
        )}
      </div>
    </nav>
  );
}
