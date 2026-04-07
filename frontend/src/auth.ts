export const TOKEN_KEY = 'xinyu_token';
export const ROLE_KEY = 'xinyu_role';

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getRole(): 'visitor' | 'counselor' | null {
  return localStorage.getItem(ROLE_KEY) as 'visitor' | 'counselor' | null;
}

export function saveAuth(token: string, role: string): void {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(ROLE_KEY, role);
}

export function clearAuth(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(ROLE_KEY);
}

export function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
