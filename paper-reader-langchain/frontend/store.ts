/** 全局状态：文档库 / 当前文档 / 对话 / 笔记 / 长期记忆 / 视图切换（Zustand）。 */

import { create } from 'zustand'
import { api, Conversation, Doc, Message, NoteIssue, streamChat, streamVision } from './api'

export type View = 'welcome' | 'catalog' | 'discovery' | 'pdf' | 'notes'
export type CatalogFilter = 'all' | 'unread' | 'read'

interface State {
  view: View
  catalogFilter: CatalogFilter
  docs: Doc[]
  catalog: Doc[]
  currentDoc: Doc | null
  conversations: Conversation[]
  messages: Message[]
  conversationId: number | null
  quote: string
  chatWarning: string
  noteIssues: NoteIssue[]
  streaming: boolean
  streamKind: 'chat' | 'vision' | null
  setView: (v: View) => void
  setCatalogFilter: (f: CatalogFilter) => void
  setQuote: (q: string) => void
  clearQuote: () => void
  clearChatWarning: () => void
  refreshDocs: () => Promise<void>
  refreshCatalog: () => Promise<void>
  refreshConversations: () => Promise<void>
  openDoc: (doc: Doc) => Promise<void>
  openConversation: (conv: Conversation) => Promise<void>
  newChat: () => void
  sendMessage: (text: string) => Promise<void>
  interpretPage: (page: number) => Promise<void>
  markRead: (doc: Doc, read: boolean) => Promise<void>
  deleteDoc: (doc: Doc) => Promise<void>
  applyNoteIssue: (id: string) => Promise<void>
  dismissNoteIssue: (id: string) => void
}

/** 更新消息列表中最后一条（流式追加用）。 */
function updateLast(messages: Message[], updater: (m: Message) => Message): Message[] {
  if (messages.length === 0) return messages
  const next = messages.slice()
  next[next.length - 1] = updater(next[next.length - 1])
  return next
}

/** 在笔记正文中定位原句并替换为修正块（含勘误留痕）；精确与空白归一两轮匹配均失败返回 null。 */
function replaceOnce(content: string, original: string, correction: string): string | null {
  const stamp = new Date().toISOString().slice(0, 10)
  const block = `${correction}\n\n> 勘误（${stamp}）：原句「${original}」`
  if (content.includes(original)) return content.replace(original, () => block)
  // 宽松匹配：忽略空白差异（Agent 摘录可能不完全逐字一致）
  const esc = (t: string) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const pattern = original.trim().split(/\s+/).filter(Boolean).map(esc).join('\\s+')
  if (!pattern) return null
  const re = new RegExp(pattern)
  return re.test(content) ? content.replace(re, () => block) : null
}

export const useStore = create<State>((set, get) => {
  /** 追加文本到最后一条消息（流式输出与错误提示共用）。 */
  const appendDelta = (text: string) =>
    set((s) => ({ messages: updateLast(s.messages, (m) => ({ ...m, content: m.content + text })) }))

  /** 开始一轮问答：插入用户消息与助手占位消息，清空选中引用与轨迹，进入流式状态。 */
  const beginTurn = (userText: string, selectedText = '', kind: 'chat' | 'vision' = 'chat') => {
    const { conversationId, messages } = get()
    const now = new Date().toISOString()
    const userMsg: Message = {
      id: -1, conversation_id: conversationId ?? -1, role: 'user',
      content: userText, selected_text: selectedText, created_at: now,
    }
    const botMsg: Message = {
      id: -2, conversation_id: conversationId ?? -1, role: 'assistant',
      content: '', selected_text: '', created_at: now,
    }
    set({ messages: [...messages, userMsg, botMsg], streaming: true, streamKind: kind, quote: '', chatWarning: '' })
  }

  return {
    view: 'welcome',
    catalogFilter: 'all',
    docs: [],
    catalog: [],
    currentDoc: null,
    conversations: [],
    messages: [],
    conversationId: null,
    quote: '',
    chatWarning: '',
    noteIssues: [],
    streaming: false,
    streamKind: null,

    setView: (v) => set({ view: v }),
    setCatalogFilter: (f) => set({ catalogFilter: f, view: 'catalog' }),
    setQuote: (q) => set({ quote: q }),
    clearQuote: () => set({ quote: '' }),
    clearChatWarning: () => set({ chatWarning: '' }),

    refreshDocs: async () => {
      set({ docs: await api.docs() })
    },
    refreshCatalog: async () => {
      set({ catalog: await api.catalog() })
    },
    refreshConversations: async () => {
      set({ conversations: await api.conversations() })
    },

    openDoc: async (doc) => {
      set({ currentDoc: doc, view: 'pdf', conversationId: null, messages: [], quote: '', chatWarning: '', noteIssues: [] })
    },

    openConversation: async (conv) => {
      const doc = get().docs.find((d) => d.id === conv.doc_id) ?? (await api.doc(conv.doc_id))
      set({ currentDoc: doc, view: 'pdf', conversationId: conv.id, quote: '', chatWarning: '', noteIssues: [] })
      set({ messages: await api.messages(conv.id) })
    },

    newChat: () => set({ conversationId: null, messages: [], quote: '', chatWarning: '', noteIssues: [] }),

    interpretPage: async (page) => {
      const { currentDoc, conversationId, streaming } = get()
      if (!currentDoc || streaming) return
      beginTurn(`请解读第 ${page} 页的图/表`, '', 'vision')
      try {
        await streamVision(
          { doc_id: currentDoc.id, page, conversation_id: conversationId ?? undefined },
          (evt) => {
            if (evt.type === 'meta') set({ conversationId: evt.conversation_id })
            else if (evt.type === 'delta') appendDelta(evt.text)
          },
        )
      } catch (e) {
        appendDelta(`\n\n> 解读失败：${(e as Error).message}`)
      } finally {
        set({ streaming: false, streamKind: null })
        await get().refreshConversations()
      }
    },

    sendMessage: async (text) => {
      const { currentDoc, conversationId, quote, streaming } = get()
      if (!currentDoc || streaming || !text.trim()) return
      beginTurn(text.trim(), quote)
      try {
        await streamChat(
          { doc_id: currentDoc.id, message: text.trim(), selected_text: quote || undefined, conversation_id: conversationId ?? undefined },
          (evt) => {
            if (evt.type === 'meta') set({ conversationId: evt.conversation_id })
            else if (evt.type === 'delta') appendDelta(evt.text)
            else if (evt.type === 'retract') set((s) => ({
              messages: updateLast(s.messages, (m) => ({
                ...m,
                content: evt.text && m.content.endsWith(evt.text) ? m.content.slice(0, -evt.text.length) : m.content,
              })),
            }))
            else if (evt.type === 'error') appendDelta(`\n\n> 出错：${evt.message}`)
            else if (evt.type === 'done') set({ chatWarning: evt.warning ?? '' })
            else if (evt.type === 'note_issue') set((s) => ({
              noteIssues: [...s.noteIssues, {
                id: `ni-${Date.now()}-${s.noteIssues.length}`, arxiv_id: evt.arxiv_id,
                original_text: evt.original_text, correction: evt.correction,
                reason: evt.reason, evidence_page: evt.evidence_page, status: 'pending',
              }],
            }))
          },
        )
      } catch (e) {
        appendDelta(`\n\n> 连接失败：${(e as Error).message}`)
      } finally {
        set({ streaming: false, streamKind: null })
        await get().refreshConversations()
        await get().refreshDocs()
        await get().refreshCatalog()
      }
    },

    markRead: async (doc, read) => {
      const updated = await api.markRead(doc.id, read)
      set((s) => ({
        currentDoc: s.currentDoc?.id === doc.id ? updated : s.currentDoc,
      }))
      await get().refreshDocs()
      await get().refreshCatalog()
    },

    deleteDoc: async (doc) => {
      await api.deleteDoc(doc.id)
      if (get().currentDoc?.id === doc.id) {
        set({ currentDoc: null, messages: [], conversationId: null, view: 'catalog' })
      }
      await get().refreshDocs()
      await get().refreshCatalog()
      await get().refreshConversations()
    },

    /** 应用勘误：拉取笔记全文 → 替换原句（含留痕）→ 经现有保存通道写回。 */
    applyNoteIssue: async (id) => {
      const issue = get().noteIssues.find((x) => x.id === id)
      if (!issue || issue.status !== 'pending') return
      const mark = (patch: Partial<NoteIssue>) =>
        set((s) => ({ noteIssues: s.noteIssues.map((x) => (x.id === id ? { ...x, ...patch } : x)) }))
      try {
        const note = await api.getNote(issue.arxiv_id)
        const next = replaceOnce(note.content || '', issue.original_text, issue.correction)
        if (next === null) throw new Error('未能在笔记中定位原句，请到「笔记」中手动修改')
        await api.updateNote(issue.arxiv_id, next)
        mark({ status: 'applied', error: undefined })
      } catch (e) {
        mark({ error: (e as Error).message })
      }
    },

    dismissNoteIssue: (id) =>
      set((s) => ({ noteIssues: s.noteIssues.map((x) => (x.id === id ? { ...x, status: 'dismissed' } : x)) })),
  }
})
