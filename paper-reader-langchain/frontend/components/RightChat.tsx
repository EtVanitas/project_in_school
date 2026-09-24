/** 右栏：对话窗（SSE 流式）。 */

import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useStore } from '../store'
import { api } from '../api'
import NoteIssueCard from './NoteIssueCard'

export default function RightChat() {
  const {
    currentDoc, messages, streaming, streamKind, quote, clearQuote, conversationId, newChat,
    chatWarning, clearChatWarning, sendMessage, noteIssues,
  } = useStore()
  const [input, setInput] = useState('')
  const [summarizing, setSummarizing] = useState(false)
  const [notice, setNotice] = useState('')
  const listRef = useRef<HTMLDivElement>(null)

  // 新消息时滚动到底部
  useEffect(() => {
    const el = listRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const submit = () => {
    if (!input.trim() || streaming || !currentDoc) return
    sendMessage(input)
    setInput('')
  }

  /** 总结整理：压缩当前对话 → 追加到该论文笔记（下次讨论自动作为背景参考）。 */
  const handleSummarize = async () => {
    if (!conversationId || summarizing || streaming) return
    setSummarizing(true)
    setNotice('')
    try {
      const r = await api.summarizeConversation(conversationId)
      setNotice(`已整理到《${r.title}》的笔记（可在「笔记」中查看编辑）`)
      setTimeout(() => setNotice(''), 5000)
    } catch (e) {
      setNotice(`整理失败：${(e as Error).message}`)
    } finally {
      setSummarizing(false)
    }
  }

  return (
    <aside className="w-[400px] shrink-0 bg-white border-l border-gray-200 flex flex-col">
      {/* 头部 */}
      <div className="h-12 shrink-0 border-b border-gray-200 flex items-center gap-2 px-3">
        <span className="text-sm font-semibold text-gray-700">AI 问答</span>
        {currentDoc && <span className="text-xs text-gray-400 truncate flex-1">{currentDoc.title}</span>}
        <div className="flex gap-1.5 shrink-0">
          <button
            onClick={handleSummarize}
            disabled={!conversationId || streaming || summarizing || messages.length < 2}
            className="text-xs px-2 py-1 border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-40"
            title="把当前对话总结成笔记，下次讨论自动参考"
          >
            {summarizing ? '整理中…' : '整理记忆'}
          </button>
          <button
            onClick={newChat}
            disabled={!currentDoc}
            className="text-xs px-2 py-1 border border-gray-300 rounded hover:bg-gray-100 disabled:opacity-40"
            title="开始新对话（保留历史记录）"
          >
            新对话
          </button>
        </div>
      </div>

      {/* 整理提示 */}
      {notice && (
        <div className="mx-3 mt-2 flex items-start gap-2 bg-emerald-50 border border-emerald-200 rounded-lg px-2.5 py-1.5">
          <div className="text-[11px] text-emerald-800 flex-1 leading-snug">{notice}</div>
          <button onClick={() => setNotice('')} className="text-emerald-400 hover:text-emerald-700 text-xs shrink-0">✕</button>
        </div>
      )}

      {/* 消息区 */}
      <div ref={listRef} className="flex-1 overflow-y-auto px-3 py-3 space-y-3 min-h-0">
        {!currentDoc && (
          <div className="text-center text-gray-400 text-sm mt-20 px-6 leading-relaxed">
            在左侧选择一篇论文开始阅读<br />选中 PDF 中的文字可直接引用提问
          </div>
        )}
        {currentDoc && messages.length === 0 && (
          <div className="text-center text-gray-400 text-sm mt-20 px-6 leading-relaxed">
            已打开《{currentDoc.title}》<br />
            可以问：这篇论文的核心贡献是什么？
          </div>
        )}
        {messages.map((m, i) => (
          <div key={m.id < 0 ? `tmp-${i}` : m.id} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div className={`max-w-[92%] rounded-xl px-3 py-2 ${m.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'}`}>
              {m.selected_text && m.role === 'user' && (
                <div className="text-[11px] bg-blue-500/40 border-l-2 border-white/70 pl-2 mb-1.5 line-clamp-2">
                  引用：{m.selected_text}
                </div>
              )}
              {m.role === 'user' ? (
                <div className="text-[13px] whitespace-pre-wrap leading-relaxed">{m.content}</div>
              ) : m.content ? (
                <div className="md"><ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown></div>
              ) : (
                <div className="text-xs text-gray-400 animate-pulse">思考中…</div>
              )}
            </div>
          </div>
        ))}
        {noteIssues.filter((x) => x.status !== 'dismissed').map((issue) => (
          <NoteIssueCard key={issue.id} issue={issue} />
        ))}
      </div>

      {/* 引用条 */}
      {quote && (
        <div className="mx-3 mb-1 flex items-start gap-2 bg-blue-50 border border-blue-200 rounded-lg px-2.5 py-1.5">
          <div className="text-[11px] text-blue-800 line-clamp-3 flex-1 leading-snug">引用：{quote}</div>
          <button onClick={clearQuote} className="text-blue-400 hover:text-blue-700 text-xs shrink-0">✕</button>
        </div>
      )}

      {/* 引用校验警告 */}
      {chatWarning && (
        <div className="mx-3 mb-1 flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1.5">
          <div className="text-[11px] text-amber-800 flex-1 leading-snug">⚠️ {chatWarning}</div>
          <button onClick={clearChatWarning} className="text-amber-400 hover:text-amber-700 text-xs shrink-0">✕</button>
        </div>
      )}

      {/* 输入区 */}
      <div className="shrink-0 border-t border-gray-200 p-3">
        <div className="flex gap-2">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            placeholder={currentDoc ? '输入问题，Enter 发送，Shift+Enter 换行' : '请先选择一篇论文'}
            disabled={!currentDoc || streaming}
            rows={2}
            className="flex-1 text-[13px] border border-gray-300 rounded-lg px-2.5 py-2 resize-none focus:outline-none focus:border-blue-500 disabled:bg-gray-50"
          />
          <button
            onClick={submit}
            disabled={!currentDoc || streaming || !input.trim()}
            className="self-end px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-40"
          >
            {streaming && streamKind === 'chat' ? '回答中' : '发送'}
          </button>
        </div>
      </div>
    </aside>
  )
}
