import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import styles from './EmotionChart.module.css';

const EMOTIONS = ['neutral', 'anxiety', 'sadness', 'anger', 'fear', 'shame', 'hopelessness'];

const EMOTION_COLORS: Record<string, string> = {
  neutral: '#94a3b8',
  anxiety: '#f59e0b',
  sadness: '#3b82f6',
  anger: '#ef4444',
  fear: '#8b5cf6',
  shame: '#ec4899',
  hopelessness: '#6b7280',
};

const EMOTION_ZH: Record<string, string> = {
  neutral: '平静', anxiety: '焦虑', sadness: '悲伤',
  anger: '愤怒', fear: '恐惧', shame: '羞耻', hopelessness: '绝望',
};

export interface ChartPoint {
  turn: number;
  intensity: number;
  emotion: string;
}

interface Props {
  data: ChartPoint[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomDot(props: any) {
  const { cx, cy, payload } = props;
  const color = EMOTION_COLORS[payload.emotion] ?? '#94a3b8';
  return <circle cx={cx} cy={cy} r={5} fill={color} stroke="#fff" strokeWidth={1.5} />;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as ChartPoint;
  return (
    <div className={styles.tooltip}>
      <div>第 {d.turn} 轮</div>
      <div>情绪：{EMOTION_ZH[d.emotion] ?? d.emotion}</div>
      <div>强度：{(d.intensity * 100).toFixed(0)}%</div>
    </div>
  );
}

export function EmotionChart({ data }: Props) {
  if (data.length < 1) {
    return <p className={styles.empty}>暂无情绪数据</p>;
  }
  return (
    <div className={styles.wrapper}>
      <div className={styles.title}>情绪强度趋势</div>
      <div className={styles.legend}>
        {EMOTIONS.map(e => (
          <span key={e} className={styles.legendItem}>
            <span className={styles.dot} style={{ background: EMOTION_COLORS[e] }} />
            {EMOTION_ZH[e]}
          </span>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="turn" label={{ value: '轮次', position: 'insideBottomRight', offset: -4, fontSize: 11 }} tick={{ fontSize: 11 }} />
          <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="intensity"
            stroke="#64748b"
            strokeWidth={2}
            dot={<CustomDot />}
            activeDot={{ r: 6 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
