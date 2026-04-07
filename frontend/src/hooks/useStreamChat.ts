import type { AlertPayload, CompletePayload, MetaPayload } from '../types';
import { authHeaders } from '../auth';

interface StreamCallbacks {
  onMeta: (meta: MetaPayload) => void;
  onAlert: (alert: AlertPayload) => void;
  onToken: (text: string) => void;
  onComplete: (complete: CompletePayload) => void;
  onError: (err: Error) => void;
}

interface SseEvent {
  event: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  data: any;
}

function parseSseChunk(buffer: string): { events: SseEvent[]; remainder: string } {
  // sse-starlette uses CRLF; normalise to LF before splitting.
  const normalised = buffer.replace(/\r\n/g, '\n');
  const blocks = normalised.split('\n\n');
  const remainder = blocks.pop() ?? '';
  const events: SseEvent[] = [];
  for (const block of blocks) {
    let eventType = '';
    let dataStr = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) eventType = line.slice(6).trim();
      if (line.startsWith('data:')) dataStr = line.slice(5).trim();
    }
    if (eventType && dataStr) {
      try {
        events.push({ event: eventType, data: JSON.parse(dataStr) });
      } catch {
        // skip malformed JSON
      }
    }
  }
  return { events, remainder };
}

export async function streamChat(
  sessionId: string,
  message: string,
  callbacks: StreamCallbacks,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`/api/v1/chat/${sessionId}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify({ message }),
    });
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
    return;
  }

  if (!response.ok) {
    callbacks.onError(new Error(`HTTP ${response.status}`));
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    callbacks.onError(new Error('No response body'));
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, remainder } = parseSseChunk(buffer);
      buffer = remainder;
      for (const evt of events) {
        if (evt.event === 'meta') callbacks.onMeta(evt.data as MetaPayload);
        else if (evt.event === 'alert') callbacks.onAlert(evt.data as AlertPayload);
        else if (evt.event === 'token') callbacks.onToken((evt.data as { text: string }).text);
        else if (evt.event === 'complete') callbacks.onComplete(evt.data as CompletePayload);
      }
    }
  } catch (err) {
    callbacks.onError(err instanceof Error ? err : new Error(String(err)));
  } finally {
    reader.releaseLock();
  }
}
