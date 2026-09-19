/** 后端 API 客户端：REST 封装 + SSE（fetch 流）解析 + 类型定义。 */

export interface Doc {
  id: number
  source: string
  arxiv_id: string
  title: string
  authors: string
  abstract: string
  published: string
  url: string
  pdf_path: string
  status: 'unread' | 'read'
  created_at: string
  read_at: string | null
  note_count?: number
  conversation_count?: number
}

export interface Candidate {
  source: string
  arxiv_id: string
  title: string
  authors: string
  abstract: string
  published: string
  url: string
  upvotes: number
  /** 个性化推荐命中词（排行榜前 3 且命中时非空；null/缺省 = 不推荐）。 */
  reco?: string[] | null
}

/** POST /api/discover 返回：候选列表 + 抓取兜底信息（stale 时展示本地缓存）。 */
export interface DiscoverResult {
  items: Candidate[]
  stale: boolean
  cached_at: string
  period: string
}

export interface Message {
  id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string
  selected_text: string
  created_at: string
}

export interface Conversation {
  id: number
  doc_id: number
  title: string
  created_at: string
  doc_title: string
}

export interface Note {
  arxiv_id: string
  title: string
  content: string
  metadata?: Record<string, any>
  updated_at?: string
}

export interface DownloadResult {
  downloaded: string[]
  skipped: string[]
  errors: string[]
}


async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const resp = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!resp.ok) {
    let detail = `请求失败（${resp.status}）`
    try {
      const body = await resp.json()
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* 保留默认消息 */ }
    throw new Error(detail)
  }
  return resp.json() as Promise<T>
}

export const api = {
  discover: (source: 'hf_papers' | 'arxiv', query = '', limit = 20, period = 'daily') =>
    request<DiscoverResult>('/api/discover', { method: 'POST', body: JSON.stringify({ source, query, limit, period }) }),
  download: (payload: { items?: Partial<Candidate>[]; arxiv_id?: string }) =>
    request<DownloadResult>('/api/download', { method: 'POST', body: JSON.stringify(payload) }),
  docs: (status?: 'unread' | 'read') =>
    request<Doc[]>(`/api/docs${status ? `?status=${status}` : ''}`),
  catalog: () => request<Doc[]>('/api/catalog'),
  doc: (id: number) => request<Doc>(`/api/docs/${id}`),
  markRead: (id: number, read: boolean) =>
    request<Doc>(`/api/docs/${id}/${read ? 'mark_read' : 'mark_unread'}`, { method: 'POST' }),
  deleteDoc: (id: number) => request<{ message: string }>(`/api/docs/${id}`, { method: 'DELETE' }),
  conversations: (docId?: number) =>
    request<Conversation[]>(`/api/conversations${docId ? `?doc_id=${docId}` : ''}`),
  messages: (convId: number) => request<Message[]>(`/api/conversations/${convId}/messages`),
  notes: (docId?: number) => request<Note[]>(`/api/notes${docId ? `?doc_id=${docId}` : ''}`),
  getNote: (arxivId: string) => request<Note>(`/api/notes/${arxivId}`),
  updateNote: (arxivId: string, content: string) =>
    request<Note>(`/api/notes/${arxivId}`, { method: 'POST', body: JSON.stringify({ content }) }),
  deleteNote: (arxivId: string) => request<{ message: string }>(`/api/notes/${arxivId}`, { method: 'DELETE' }),
  /** 总结整理：把整段对话压缩为结构化笔记并保存到该论文（可编辑）。 */
  summarizeConversation: (convId: number) =>
    request<{ summary: string; arxiv_id: string; title: string }>(
      `/api/conversations/${convId}/summarize`,
      { method: 'POST' },
    ),
}

export type ChatEvent =
  | { type: 'meta'; conversation_id: number; doc_id: number }
  | { type: 'agent_step'; step: number; tool: string; args_summary: string }
  | { type: 'agent_observation'; step: number; tool: string; ok: boolean; summary: string }
  | { type: 'note_issue'; step: number; arxiv_id: string; original_text: string; correction: string; reason: string; evidence_page: number }
  | { type: 'delta'; text: string }
  | { type: 'retract'; text: string }
  | { type: 'done'; message_id: number; conversation_id: number; warning?: string }
  | { type: 'error'; message: string }

/** 笔记勘误建议（Agent 核实笔记与论文矛盾后提交；用户确认后经现有保存通道更正）。 */
export interface NoteIssue {
  id: string
  arxiv_id: string
  original_text: string
  correction: string
  reason: string
  evidence_page: number
  status: 'pending' | 'applied' | 'dismissed'
  error?: string
}

/** 图表解读事件（POST /api/vision_page）。 */
export type VisionEvent =
  | { type: 'meta'; conversation_id: number; doc_id: number }
  | { type: 'delta'; text: string }
  | { type: 'done'; message_id: number; conversation_id: number }

/** SSE 通用解析（fetch + ReadableStream 逐事件回调）。 */
async function streamSSE<T>(url: string, body: unknown, onEvent: (evt: T) => void): Promise<void> {
  const resp = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!resp.ok || !resp.body) {
    let detail = `请求失败（${resp.status}）`
    try { const b = await resp.json(); if (b?.detail) detail = String(b.detail) } catch { /* 默认消息 */ }
    throw new Error(detail)
  }
  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) {
          try { onEvent(JSON.parse(line.slice(6)) as T) } catch { /* 忽略坏帧 */ }
        }
      }
    }
  }
}

/** SSE 流式问答：逐事件回调。 */
export function streamChat(
  body: { doc_id: number; message: string; selected_text?: string; conversation_id?: number },
  onEvent: (evt: ChatEvent) => void,
): Promise<void> {
  return streamSSE('/api/chat', body, onEvent)
}

/** SSE 图表解读：meta / delta / done。 */
export function streamVision(
  body: { doc_id: number; page: number; conversation_id?: number },
  onEvent: (evt: VisionEvent) => void,
): Promise<void> {
  return streamSSE('/api/vision_page', body, onEvent)
}
