import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import styles from './DashboardCharts.module.css';

export interface EmotionCount { emotion: string; count: number; }
export interface RiskLevelCount { risk_level: string; count: number; }

interface Props {
  emotionData: EmotionCount[];
  riskData: RiskLevelCount[];
}

const EMOTION_ZH: Record<string, string> = {
  neutral: '平静', anxiety: '焦虑', sadness: '悲伤',
  anger: '愤怒', fear: '恐惧', shame: '羞耻', hopelessness: '绝望',
};

const EMOTION_COLORS: Record<string, string> = {
  neutral: '#94a3b8', anxiety: '#f59e0b', sadness: '#3b82f6',
  anger: '#ef4444', fear: '#8b5cf6', shame: '#ec4899', hopelessness: '#6b7280',
};

const RISK_COLORS: Record<string, string> = {
  L0: '#22c55e', L1: '#f59e0b', L2: '#ef4444', L3: '#7f1d1d',
};

const RISK_ZH: Record<string, string> = {
  L0: 'L0 正常', L1: 'L1 关注', L2: 'L2 警告', L3: 'L3 危机',
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function PieLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) {
  if (percent < 0.05) return null;
  const RADIAN = Math.PI / 180;
  const r = innerRadius + (outerRadius - innerRadius) * 0.55;
  const x = cx + r * Math.cos(-midAngle * RADIAN);
  const y = cy + r * Math.sin(-midAngle * RADIAN);
  return (
    <text x={x} y={y} fill="#fff" textAnchor="middle" dominantBaseline="central" fontSize={11} fontWeight={600}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  );
}

export function DashboardCharts({ emotionData, riskData }: Props) {
  const pieData = emotionData.map(d => ({
    name: EMOTION_ZH[d.emotion] ?? d.emotion,
    value: d.count,
    emotion: d.emotion,
  }));

  const barData = riskData.map(d => ({
    name: RISK_ZH[d.risk_level] ?? d.risk_level,
    count: d.count,
    risk_level: d.risk_level,
  }));

  const hasEmotionData = pieData.length > 0;
  const hasRiskData = barData.length > 0;

  return (
    <div className={styles.row}>
      {/* Emotion distribution pie chart */}
      <div className={styles.card}>
        <div className={styles.title}>情绪分布</div>
        {hasEmotionData ? (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                labelLine={false}
                label={PieLabel}
              >
                {pieData.map(entry => (
                  <Cell
                    key={entry.emotion}
                    fill={EMOTION_COLORS[entry.emotion] ?? '#94a3b8'}
                  />
                ))}
              </Pie>
              <Tooltip
                formatter={(value, name) => [String(value) + ' 条', String(name)]}
              />
              <Legend
                iconType="circle"
                iconSize={8}
                wrapperStyle={{ fontSize: '0.78rem' }}
              />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <p className={styles.empty}>暂无数据</p>
        )}
      </div>

      {/* Risk level distribution bar chart */}
      <div className={styles.card}>
        <div className={styles.title}>风险等级分布（会话维度）</div>
        {hasRiskData ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={barData} margin={{ top: 8, right: 16, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value) => [String(value) + ' 个会话', '会话数']} />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {barData.map(entry => (
                  <Cell
                    key={entry.risk_level}
                    fill={RISK_COLORS[entry.risk_level] ?? '#94a3b8'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className={styles.empty}>暂无数据</p>
        )}
      </div>
    </div>
  );
}
