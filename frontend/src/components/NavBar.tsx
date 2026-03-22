import { Link, useLocation, useNavigate } from 'react-router-dom';
import { COUNSELOR_STORAGE_KEY } from './RequireAuth';
import styles from './NavBar.module.css';

export function NavBar() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const authed = sessionStorage.getItem(COUNSELOR_STORAGE_KEY) === '1';

  function handleLogout() {
    sessionStorage.removeItem(COUNSELOR_STORAGE_KEY);
    navigate('/');
  }

  return (
    <nav className={styles.nav}>
      <span className={styles.brand}>心语 XinYu</span>
      <div className={styles.links}>
        <Link className={pathname === '/' ? styles.active : ''} to="/">对话</Link>
        {authed ? (
          <>
            <Link
              className={pathname.startsWith('/dashboard') || pathname.startsWith('/session') ? styles.active : ''}
              to="/dashboard"
            >
              咨询师后台
            </Link>
            <button className={styles.logoutBtn} onClick={handleLogout}>退出</button>
          </>
        ) : null}
      </div>
    </nav>
  );
}
