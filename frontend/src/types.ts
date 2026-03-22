export type RiskLevel = 'L0' | 'L1' | 'L2' | 'L3';
export type AlertStatus = 'open' | 'acknowledged' | 'resolved';

export interface MetaPayload {
  emotion: string;
  intent: string;
  intensity: number;
  risk_level: RiskLevel;
}

export interface AlertResource {
  title: string;
  phone?: string;
  url?: string;
  description?: string;
}

export interface AlertPayload {
  risk_level: RiskLevel;
  resources: AlertResource[];
}

export interface CompletePayload {
  message_id: string;
  latency_ms: number;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  meta?: MetaPayload;
  alert?: AlertPayload;
  streaming?: boolean;
}

export interface SessionSummary {
  session_id: string;
  visitor_id: string;
  latest_risk_level: RiskLevel;
  started_at: string;
  message_count: number;
  dominant_emotion?: string;
}

export interface DashboardStats {
  total_sessions: number;
  total_messages: number;
  open_alerts: number;
  l3_alerts: number;
}

export interface AnalysisSummary {
  emotion_label: string;
  intent_label: string;
  intensity_score: number;
  risk_score: number;
}

export interface MessageWithAnalysis {
  message_id: string;
  role: string;
  content: string;
  sequence_no: number;
  safety_mode: string;
  created_at: string;
  analysis: AnalysisSummary | null;
}

export interface AlertSummary {
  alert_id: string;
  session_id: string;
  risk_level: RiskLevel;
  reasons: string[];
  status: AlertStatus;
  created_at: string;
}

export interface UserSession {
  session_id: string;
  started_at: string;
  message_count: number;
  latest_risk_level: RiskLevel;
}
