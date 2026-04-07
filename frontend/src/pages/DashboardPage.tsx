import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import type {
  AlertStatus, AlertSummary, CounselorSummary,
  DashboardStats, MeInfo, SessionSummary, VisitorSummary,
} from '../types';
import { RiskBadge } from '../components/RiskBadge';
import { DashboardCharts } from '../components/DashboardCharts';
import type { EmotionCount, RiskLevelCount } from '../components/DashboardCharts';
import { FilterBar } from '../components/FilterBar';
import type { FilterField } from '../components/FilterBar';
import { authHeaders } from '../auth';
import { COLLEGES } from '../constants';
import styles from './DashboardPage.module.css';

function formatDate(iso: string) {
  return new Date(iso).toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

const EMOTION_ZH: Record<string, string> = {
  neutral: '平静', anxiety: '焦虑', sadness: '悲伤',
  anger: '愤怒', fear: '恐惧', shame: '羞耻', hopelessness: '绝望',
};

const RISK_ORD: Record<string, number> = { L0: 0, L1: 1, L2: 2, L3: 3 };
const STATUS_ORD: Record<string, number> = { open: 0, acknowledged: 1, resolved: 2 };

function visitorDisplayName(s: SessionSummary): string {
  if (s.visitor_is_guest) return '匿名访客';
  return s.visitor_real_name || s.visitor_username || '—';
}

type Tab = 'sessions' | 'alerts' | 'visitors' | 'admin';
type SortDir = 'asc' | 'desc';
interface SortState { key: string; dir: SortDir; }

interface SessionFilters { search: string; college: string; dateFrom: string; dateTo: string; riskLevel: string; [k: string]: string; }
interface AlertFilters { status: string; riskLevel: string; dateFrom: string; dateTo: string; [k: string]: string; }
interface VisitorFilters { search: string; college: string; [k: string]: string; }

const DEFAULT_SESSION_FILTERS: SessionFilters = { search: '', college: '', dateFrom: '', dateTo: '', riskLevel: '' };
const DEFAULT_ALERT_FILTERS: AlertFilters = { status: '', riskLevel: '', dateFrom: '', dateTo: '' };
const DEFAULT_VISITOR_FILTERS: VisitorFilters = { search: '', college: '' };

function hasActiveFilters(f: Record<string, string>) {
  return Object.values(f).some(v => v !== '');
}

// Field configs for FilterBar
const COLLEGE_OPTIONS = [
  { value: '', label: '全部学院' },
  ...COLLEGES.map(c => ({ value: c, label: c })),
];
const RISK_OPTIONS = [
  { value: '', label: '全部风险' },
  { value: 'L0', label: 'L0 正常' },
  { value: 'L1', label: 'L1 关注' },
  { value: 'L2', label: 'L2 预警' },
  { value: 'L3', label: 'L3 危机' },
];
// Alert risk: L1/L2/L3 only (L0 is not alertable)
const ALERT_RISK_OPTIONS = RISK_OPTIONS.filter(o => o.value !== 'L0');

const SESSION_FIELDS: FilterField[] = [
  { type: 'text', key: 'search', placeholder: '姓名 / 学号 / 用户名' },
  { type: 'select', key: 'college', options: COLLEGE_OPTIONS },
  { type: 'select', key: 'riskLevel', options: RISK_OPTIONS },
  { type: 'date', key: 'dateFrom' },
  { type: 'date', key: 'dateTo' },
];
const ALERT_FIELDS: FilterField[] = [
  { type: 'select', key: 'status', options: [
    { value: '', label: '全部状态' },
    { value: 'open', label: '待处理' },
    { value: 'acknowledged', label: '已确认' },
    { value: 'resolved', label: '已解决' },
  ]},
  { type: 'select', key: 'riskLevel', options: ALERT_RISK_OPTIONS },
  { type: 'date', key: 'dateFrom' },
  { type: 'date', key: 'dateTo' },
];
const VISITOR_FIELDS: FilterField[] = [
  { type: 'text', key: 'search', placeholder: '姓名 / 学号 / 用户名' },
  { type: 'select', key: 'college', options: COLLEGE_OPTIONS },
];

// SortHeader helper component
function SortHeader({ label, sortKey, sort, onSort }: {
  label: string; sortKey: string; sort: SortState; onSort: (key: string) => void;
}) {
  const active = sort.key === sortKey;
  return (
    <th
      className={`${styles.thSortable}${active ? ' ' + styles.thActive : ''}`}
      onClick={() => onSort(sortKey)}
    >
      {label}<span className={styles.sortArrow}>{active ? (sort.dir === 'asc' ? ' ▲' : ' ▼') : ' ⇅'}</span>
    </th>
  );
}

export function DashboardPage() {
  const [tab, setTab] = useState<Tab>('sessions');
  const [me, setMe] = useState<MeInfo | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [alerts, setAlerts] = useState<AlertSummary[]>([]);
  const [visitors, setVisitors] = useState<VisitorSummary[]>([]);
  const [counselors, setCounselors] = useState<CounselorSummary[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [emotionDist, setEmotionDist] = useState<EmotionCount[]>([]);
  const [riskDist, setRiskDist] = useState<RiskLevelCount[]>([]);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);

  // Filter state
  const [sessionFilters, setSessionFilters] = useState<SessionFilters>(DEFAULT_SESSION_FILTERS);
  const [alertFilters, setAlertFilters] = useState<AlertFilters>(DEFAULT_ALERT_FILTERS);
  const [visitorFilters, setVisitorFilters] = useState<VisitorFilters>(DEFAULT_VISITOR_FILTERS);

  // Sort state
  const [sessionSort, setSessionSort] = useState<SortState>({ key: 'started_at', dir: 'desc' });
  const [alertSort, setAlertSort] = useState<SortState>({ key: 'status_priority', dir: 'asc' });
  const [visitorSort, setVisitorSort] = useState<SortState>({ key: 'created_at', dir: 'desc' });

  // Create counselor form state
  const [newUsername, setNewUsername] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newCollege, setNewCollege] = useState('');
  const [newDisplayName, setNewDisplayName] = useState('');
  const [createError, setCreateError] = useState('');
  const [creating, setCreating] = useState(false);

  const isSuperAdmin = me !== null && me.college === null;

  // Sort toggle helpers
  const DATE_KEYS = new Set(['started_at', 'created_at']);
  function makeToggleSort(setter: React.Dispatch<React.SetStateAction<SortState>>) {
    return (key: string) => setter(prev =>
      prev.key === key
        ? { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' }
        : { key, dir: DATE_KEYS.has(key) ? 'desc' : 'asc' }
    );
  }
  const toggleSessionSort = makeToggleSort(setSessionSort);
  const toggleAlertSort = makeToggleSort(setAlertSort);
  const toggleVisitorSort = makeToggleSort(setVisitorSort);

  // Derived: filtered + sorted sessions
  const filteredSessions = useMemo(() => {
    let result = sessions;
    const { search, college, dateFrom, dateTo, riskLevel } = sessionFilters;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(s =>
        (s.visitor_real_name ?? '').toLowerCase().includes(q)
        || (s.visitor_username ?? '').toLowerCase().includes(q)
        || (s.visitor_student_id ?? '').toLowerCase().includes(q)
      );
    }
    if (college) result = result.filter(s => s.visitor_college === college);
    if (riskLevel) result = result.filter(s => s.latest_risk_level === riskLevel);
    if (dateFrom) {
      const from = new Date(dateFrom + 'T00:00:00');
      result = result.filter(s => s.started_at && new Date(s.started_at) >= from);
    }
    if (dateTo) {
      const to = new Date(dateTo + 'T23:59:59');
      result = result.filter(s => s.started_at && new Date(s.started_at) <= to);
    }
    // Sort
    const { key, dir } = sessionSort;
    const mul = dir === 'asc' ? 1 : -1;
    return [...result].sort((a, b) => {
      if (key === 'started_at') return mul * (new Date(a.started_at).getTime() - new Date(b.started_at).getTime());
      if (key === 'latest_risk_level') return mul * ((RISK_ORD[a.latest_risk_level] ?? 0) - (RISK_ORD[b.latest_risk_level] ?? 0));
      if (key === 'visitor_real_name') return mul * (visitorDisplayName(a).localeCompare(visitorDisplayName(b), 'zh'));
      return 0;
    });
  }, [sessions, sessionFilters, sessionSort]);

  // Derived: filtered + sorted alerts
  const filteredAlerts = useMemo(() => {
    let result = alerts;
    const { status, riskLevel, dateFrom, dateTo } = alertFilters;
    if (status) result = result.filter(a => a.status === status);
    if (riskLevel) result = result.filter(a => a.risk_level === riskLevel);
    if (dateFrom) {
      const from = new Date(dateFrom + 'T00:00:00');
      result = result.filter(a => a.created_at && new Date(a.created_at) >= from);
    }
    if (dateTo) {
      const to = new Date(dateTo + 'T23:59:59');
      result = result.filter(a => a.created_at && new Date(a.created_at) <= to);
    }
    // Sort
    const { key, dir } = alertSort;
    const mul = dir === 'asc' ? 1 : -1;
    return [...result].sort((a, b) => {
      if (key === 'status_priority') {
        const sd = (STATUS_ORD[a.status] ?? 0) - (STATUS_ORD[b.status] ?? 0);
        if (sd !== 0) return sd; // pending first regardless of dir
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
      if (key === 'created_at') return mul * (new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
      if (key === 'risk_level') return mul * ((RISK_ORD[a.risk_level] ?? 0) - (RISK_ORD[b.risk_level] ?? 0));
      if (key === 'status') return mul * ((STATUS_ORD[a.status] ?? 0) - (STATUS_ORD[b.status] ?? 0));
      return 0;
    });
  }, [alerts, alertFilters, alertSort]);

  // Derived: filtered + sorted visitors
  const filteredVisitors = useMemo(() => {
    let result = visitors;
    const { search, college } = visitorFilters;
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(v =>
        (v.real_name ?? '').toLowerCase().includes(q)
        || (v.username ?? '').toLowerCase().includes(q)
        || (v.student_id ?? '').toLowerCase().includes(q)
      );
    }
    if (college) result = result.filter(v => v.college === college);
    // Sort
    const { key, dir } = visitorSort;
    const mul = dir === 'asc' ? 1 : -1;
    return [...result].sort((a, b) => {
      if (key === 'real_name') return mul * ((a.real_name ?? a.username ?? '').localeCompare(b.real_name ?? b.username ?? '', 'zh'));
      if (key === 'created_at') return mul * (new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
      if (key === 'session_count') return mul * (a.session_count - b.session_count);
      if (key === 'latest_risk_level') return mul * ((RISK_ORD[a.latest_risk_level ?? 'L0'] ?? 0) - (RISK_ORD[b.latest_risk_level ?? 'L0'] ?? 0));
      return 0;
    });
  }, [visitors, visitorFilters, visitorSort]);

  useEffect(() => {
    const h = { headers: authHeaders() };
    const fetches: Promise<unknown>[] = [
      fetch('/api/v1/auth/me', h).then(r => r.json()),
      fetch('/api/v1/dashboard/sessions', h).then(r => r.json()),
      fetch('/api/v1/dashboard/alerts', h).then(r => r.json()),
      fetch('/api/v1/dashboard/stats', h).then(r => r.json()),
      fetch('/api/v1/dashboard/charts', h).then(r => r.json()),
      fetch('/api/v1/dashboard/visitors', h).then(r => r.json()),
    ];

    Promise.all(fetches).then(([meData, sData, aData, stData, cData, vData]) => {
      setMe(meData as MeInfo);
      setSessions((sData as { sessions: SessionSummary[] }).sessions ?? []);
      setAlerts((aData as { alerts: AlertSummary[] }).alerts ?? []);
      setStats(stData as DashboardStats);
      setEmotionDist((cData as { emotion_distribution: EmotionCount[] }).emotion_distribution ?? []);
      setRiskDist((cData as { risk_distribution: RiskLevelCount[] }).risk_distribution ?? []);
      setVisitors((vData as { visitors: VisitorSummary[] }).visitors ?? []);
      setLoading(false);
    }).catch(err => {
      console.error(err);
      setLoading(false);
    });
  }, []);

  // Load counselors once we know this is a super admin
  useEffect(() => {
    if (!isSuperAdmin) return;
    fetch('/api/v1/dashboard/counselors', { headers: authHeaders() })
      .then(r => r.json())
      .then(d => setCounselors(d.counselors ?? []));
  }, [isSuperAdmin]);

  async function updateAlertStatus(alertId: string, newStatus: AlertStatus) {
    const res = await fetch(`/api/v1/dashboard/alerts/${alertId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ status: newStatus }),
    });
    if (!res.ok) return;
    const updated: AlertSummary = await res.json();
    setAlerts(prev => prev.map(a => a.alert_id === alertId ? updated : a));
  }

  async function handleExport() {
    setExporting(true);
    try {
      const params = new URLSearchParams();
      if (sessionFilters.search)    params.set('search', sessionFilters.search);
      if (sessionFilters.riskLevel) params.set('risk_level', sessionFilters.riskLevel);
      if (sessionFilters.dateFrom)  params.set('date_from', sessionFilters.dateFrom);
      if (sessionFilters.dateTo)    params.set('date_to', sessionFilters.dateTo);
      if (alertFilters.status)      params.set('alert_status', alertFilters.status);
      if (alertFilters.riskLevel)   params.set('alert_risk_level', alertFilters.riskLevel);
      if (alertFilters.dateFrom)    params.set('alert_date_from', alertFilters.dateFrom);
      if (alertFilters.dateTo)      params.set('alert_date_to', alertFilters.dateTo);

      const url = `/api/v1/dashboard/export${params.toString() ? '?' + params.toString() : ''}`;
      const res = await fetch(url, { headers: authHeaders() });
      if (!res.ok) { alert('导出失败'); return; }
      const blob = await res.blob();
      const objUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const cd = res.headers.get('Content-Disposition') ?? '';
      const match = cd.match(/filename="(.+?)"/);
      a.download = match ? match[1] : 'xinyu_export.xlsx';
      a.href = objUrl;
      a.click();
      URL.revokeObjectURL(objUrl);
    } catch {
      alert('导出失败，请重试');
    } finally {
      setExporting(false);
    }
  }

  async function handleCreateCounselor(e: React.FormEvent) {
    e.preventDefault();
    setCreateError('');
    setCreating(true);
    try {
      const res = await fetch('/api/v1/dashboard/counselors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeaders() },
        body: JSON.stringify({
          username: newUsername,
          password: newPassword,
          college: newCollege,
          display_name: newDisplayName || undefined,
        }),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        setCreateError(d.detail ?? '创建失败');
        return;
      }
      const created: CounselorSummary = await res.json();
      setCounselors(prev => [...prev, created]);
      setNewUsername(''); setNewPassword(''); setNewCollege(''); setNewDisplayName('');
    } catch {
      setCreateError('网络错误，请重试');
    } finally {
      setCreating(false);
    }
  }

  async function handleToggleActive(counselorId: string) {
    const res = await fetch(`/api/v1/dashboard/counselors/${counselorId}`, {
      method: 'PATCH',
      headers: authHeaders(),
    });
    if (!res.ok) return;
    const updated: CounselorSummary = await res.json();
    setCounselors(prev => prev.map(c => c.counselor_id === counselorId ? updated : c));
  }

  if (loading) return <div className={styles.loading}>加载中…</div>;

  return (
    <div className={styles.page}>
      {/* Stats bar */}
      {stats && (
        <div className={styles.statsBar}>
          <div className={`${styles.statCard} ${styles.statCardBlue}`}>
            <div className={styles.statNumber}>{stats.total_sessions}</div>
            <div className={styles.statLabel}>总会话数</div>
          </div>
          <div className={`${styles.statCard} ${styles.statCardPurple}`}>
            <div className={styles.statNumber}>{stats.total_messages}</div>
            <div className={styles.statLabel}>总消息数</div>
          </div>
          <div className={`${styles.statCard} ${styles.statCardAmber}`}>
            <div className={stats.open_alerts > 0 ? styles.statNumberAlert : styles.statNumber}>
              {stats.open_alerts}
            </div>
            <div className={styles.statLabel}>待处理预警</div>
          </div>
          <div className={`${styles.statCard} ${styles.statCardRed}`}>
            <div className={stats.l3_alerts > 0 ? styles.statNumberAlert : styles.statNumber}>
              {stats.l3_alerts}
            </div>
            <div className={styles.statLabel}>L3 危机预警</div>
          </div>
        </div>
      )}

      {/* Visualisation charts */}
      <section className={styles.section}>
        <h2 className={styles.heading}>数据可视化</h2>
        <DashboardCharts emotionData={emotionDist} riskData={riskDist} />
      </section>

      {/* Tabs + export button */}
      <section className={styles.section}>
        <div className={styles.tabRow}>
          <div className={styles.tabs}>
            <button
              className={tab === 'sessions' ? styles.tabActive : styles.tabBtn}
              onClick={() => setTab('sessions')}
            >
              会话记录 ({hasActiveFilters(sessionFilters) ? `${filteredSessions.length}/${sessions.length}` : sessions.length})
            </button>
            <button
              className={tab === 'alerts' ? styles.tabActive : styles.tabBtn}
              onClick={() => setTab('alerts')}
            >
              预警记录 ({hasActiveFilters(alertFilters) ? `${filteredAlerts.length}/${alerts.length}` : alerts.length})
            </button>
            <button
              className={tab === 'visitors' ? styles.tabActive : styles.tabBtn}
              onClick={() => setTab('visitors')}
            >
              学生档案 ({hasActiveFilters(visitorFilters) ? `${filteredVisitors.length}/${visitors.length}` : visitors.length})
            </button>
            {isSuperAdmin && (
              <button
                className={tab === 'admin' ? styles.tabActive : styles.tabBtn}
                onClick={() => setTab('admin')}
              >管理员设置</button>
            )}
          </div>
          <button className={styles.exportBtn} onClick={handleExport} disabled={exporting}>
            {exporting ? '导出中…' : '⬇ 导出 Excel'}
          </button>
        </div>

        {/* Sessions tab */}
        {tab === 'sessions' && (
          <>
            <FilterBar
              fields={SESSION_FIELDS}
              values={sessionFilters}
              onChange={(k, v) => setSessionFilters(p => ({ ...p, [k]: v }))}
              onClear={() => setSessionFilters(DEFAULT_SESSION_FILTERS)}
              resultCount={filteredSessions.length}
              totalCount={sessions.length}
            />
            {filteredSessions.length === 0 ? (
              <p className={styles.empty}>暂无匹配的会话记录</p>
            ) : (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th>会话 ID</th>
                    <SortHeader label="学生" sortKey="visitor_real_name" sort={sessionSort} onSort={toggleSessionSort} />
                    <th>学院</th>
                    <SortHeader label="开始时间" sortKey="started_at" sort={sessionSort} onSort={toggleSessionSort} />
                    <th>消息数</th>
                    <th>主要情绪</th>
                    <SortHeader label="最高风险" sortKey="latest_risk_level" sort={sessionSort} onSort={toggleSessionSort} />
                  </tr>
                </thead>
                <tbody>
                  {filteredSessions.map(s => (
                    <tr key={s.session_id}>
                      <td>
                        <Link to={`/session/${s.session_id}`} className={styles.link}>
                          {s.session_id.slice(0, 8)}…
                        </Link>
                      </td>
                      <td>{visitorDisplayName(s)}</td>
                      <td>{s.visitor_college ?? '—'}</td>
                      <td>{formatDate(s.started_at)}</td>
                      <td>{s.message_count}</td>
                      <td className={styles.emotionCell}>
                        {s.dominant_emotion ? (EMOTION_ZH[s.dominant_emotion] ?? s.dominant_emotion) : '—'}
                      </td>
                      <td><RiskBadge level={s.latest_risk_level} small /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}

        {/* Alerts tab */}
        {tab === 'alerts' && (
          <>
            <FilterBar
              fields={ALERT_FIELDS}
              values={alertFilters}
              onChange={(k, v) => setAlertFilters(p => ({ ...p, [k]: v }))}
              onClear={() => setAlertFilters(DEFAULT_ALERT_FILTERS)}
              resultCount={filteredAlerts.length}
              totalCount={alerts.length}
            />
            {filteredAlerts.length === 0 ? (
              <p className={styles.empty}>暂无匹配的预警记录</p>
            ) : (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <SortHeader label="时间" sortKey="created_at" sort={alertSort} onSort={toggleAlertSort} />
                    <th>会话</th>
                    <SortHeader label="风险等级" sortKey="risk_level" sort={alertSort} onSort={toggleAlertSort} />
                    <th>原因</th>
                    <SortHeader label="状态 / 操作" sortKey="status" sort={alertSort} onSort={toggleAlertSort} />
                  </tr>
                </thead>
                <tbody>
                  {filteredAlerts.map(a => (
                    <tr key={a.alert_id}>
                      <td>{formatDate(a.created_at)}</td>
                      <td>
                        <Link to={`/session/${a.session_id}`} className={styles.link}>
                          {a.session_id.slice(0, 8)}…
                        </Link>
                      </td>
                      <td><RiskBadge level={a.risk_level} small /></td>
                      <td className={styles.reasons}>{a.reasons.join('；')}</td>
                      <td className={styles.actionCell}>
                        {a.status === 'resolved' ? (
                          <span className={`${styles.status} ${styles.resolved}`}>已解决</span>
                        ) : (
                          <div className={styles.actionBtns}>
                            {a.status === 'open' && (
                              <button
                                className={`${styles.actionBtn} ${styles.actionBtnAck}`}
                                onClick={() => updateAlertStatus(a.alert_id, 'acknowledged')}
                              >确认</button>
                            )}
                            <button
                              className={`${styles.actionBtn} ${styles.actionBtnResolve}`}
                              onClick={() => updateAlertStatus(a.alert_id, 'resolved')}
                            >解决</button>
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}

        {/* Visitors tab */}
        {tab === 'visitors' && (
          <>
            <FilterBar
              fields={VISITOR_FIELDS}
              values={visitorFilters}
              onChange={(k, v) => setVisitorFilters(p => ({ ...p, [k]: v }))}
              onClear={() => setVisitorFilters(DEFAULT_VISITOR_FILTERS)}
              resultCount={filteredVisitors.length}
              totalCount={visitors.length}
            />
            {filteredVisitors.length === 0 ? (
              <p className={styles.empty}>暂无匹配的注册学生</p>
            ) : (
              <table className={styles.table}>
                <thead>
                  <tr>
                    <SortHeader label="姓名" sortKey="real_name" sort={visitorSort} onSort={toggleVisitorSort} />
                    <th>学号</th>
                    <th>学院</th>
                    <th>用户名</th>
                    <SortHeader label="注册时间" sortKey="created_at" sort={visitorSort} onSort={toggleVisitorSort} />
                    <SortHeader label="会话数" sortKey="session_count" sort={visitorSort} onSort={toggleVisitorSort} />
                    <SortHeader label="最高风险" sortKey="latest_risk_level" sort={visitorSort} onSort={toggleVisitorSort} />
                  </tr>
                </thead>
                <tbody>
                  {filteredVisitors.map(v => (
                    <tr key={v.visitor_id}>
                      <td>{v.real_name ?? '—'}</td>
                      <td>{v.student_id ?? '—'}</td>
                      <td>{v.college ?? '—'}</td>
                      <td>{v.username ?? '—'}</td>
                      <td>{formatDate(v.created_at)}</td>
                      <td>{v.session_count}</td>
                      <td>
                        {v.latest_risk_level
                          ? <RiskBadge level={v.latest_risk_level} small />
                          : <span style={{ color: '#94a3b8' }}>—</span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </>
        )}

        {/* Admin tab (super admin only) */}
        {tab === 'admin' && isSuperAdmin && (
          <div className={styles.adminPanel}>
            {/* Create counselor form */}
            <div className={styles.adminCard}>
              <h3 className={styles.adminCardTitle}>添加学院咨询师</h3>
              <form onSubmit={handleCreateCounselor} className={styles.adminForm}>
                <input
                  className={styles.adminInput}
                  type="text"
                  placeholder="用户名"
                  value={newUsername}
                  onChange={e => setNewUsername(e.target.value)}
                  required
                />
                <input
                  className={styles.adminInput}
                  type="password"
                  placeholder="密码（至少6位）"
                  value={newPassword}
                  onChange={e => setNewPassword(e.target.value)}
                  required
                  minLength={6}
                />
                <select
                  className={styles.adminSelect}
                  value={newCollege}
                  onChange={e => setNewCollege(e.target.value)}
                  required
                >
                  <option value="" disabled>请选择所属学院</option>
                  {COLLEGES.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
                <input
                  className={styles.adminInput}
                  type="text"
                  placeholder="姓名（可选）"
                  value={newDisplayName}
                  onChange={e => setNewDisplayName(e.target.value)}
                />
                {createError && <p className={styles.adminError}>{createError}</p>}
                <button
                  className={styles.adminSubmitBtn}
                  type="submit"
                  disabled={creating || !newUsername || !newPassword || !newCollege}
                >
                  {creating ? '创建中…' : '创建咨询师账户'}
                </button>
              </form>
            </div>

            {/* Counselor list */}
            <div className={styles.adminCard}>
              <h3 className={styles.adminCardTitle}>所有咨询师账户 ({counselors.length})</h3>
              {counselors.length === 0 ? (
                <p className={styles.empty}>暂无咨询师账户</p>
              ) : (
                <table className={styles.table}>
                  <thead>
                    <tr>
                      <th>用户名</th>
                      <th>姓名</th>
                      <th>所属学院</th>
                      <th>权限</th>
                      <th>状态</th>
                      <th>操作</th>
                    </tr>
                  </thead>
                  <tbody>
                    {counselors.map(c => (
                      <tr key={c.counselor_id}>
                        <td>{c.username}</td>
                        <td>{c.display_name ?? '—'}</td>
                        <td>{c.college ?? '—'}</td>
                        <td>{c.college === null ? <span className={styles.superAdminBadge}>超级管理员</span> : '学院咨询师'}</td>
                        <td>
                          <span className={`${styles.status} ${c.is_active ? styles.resolved : styles.open}`}>
                            {c.is_active ? '启用' : '停用'}
                          </span>
                        </td>
                        <td>
                          {c.college !== null && (
                            <button
                              className={`${styles.actionBtn} ${c.is_active ? styles.actionBtnAck : styles.actionBtnResolve}`}
                              onClick={() => handleToggleActive(c.counselor_id)}
                            >
                              {c.is_active ? '停用' : '启用'}
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
