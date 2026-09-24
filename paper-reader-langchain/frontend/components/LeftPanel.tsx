/** 左栏：上=对话历史，下=文档库（未读区/已读区）+ 目录/发现入口。 */

import { useStore } from '../store'
import type { Doc } from '../api'

function DocItem({ doc }: { doc: Doc }) {
  const { currentDoc, openDoc } = useStore()
  const active = currentDoc?.id === doc.id
  return (
    <button
      onClick={() => openDoc(doc)}
      title={doc.title}
      className={`w-full text-left px-2.5 py-1.5 rounded-md text-[13px] leading-snug transition-colors ${
        active ? 'bg-blue-100 text-blue-900' : 'hover:bg-gray-100 text-gray-700'
      }`}
    >
      <div className="line-clamp-2">{doc.title}</div>
      <div className="text-[11px] text-gray-400 mt-0.5">
        {doc.source === 'legacy' ? '本地' : doc.source === 'hf_papers' ? 'HF 热门' : 'arXiv'}
        {doc.arxiv_id ? ` · ${doc.arxiv_id}` : ''}
      </div>
    </button>
  )
}

export default function LeftPanel() {
  const { docs, conversations, openConversation, setView, setCatalogFilter } = useStore()
  const unread = docs.filter((d) => d.status === 'unread')
  const read = docs.filter((d) => d.status === 'read')

  return (
    <aside className="w-72 shrink-0 bg-white border-r border-gray-200 flex flex-col">
      {/* 上：对话历史 */}
      <div className="h-[38%] flex flex-col border-b border-gray-200 min-h-0">
        <div className="px-3 py-2 text-xs font-semibold text-gray-500 tracking-wide">对话历史</div>
        <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
          {conversations.length === 0 && (
            <div className="text-[12px] text-gray-400 px-2 py-1">暂无对话，打开论文后即可提问</div>
          )}
          {conversations.map((c) => (
            <button
              key={c.id}
              onClick={() => openConversation(c)}
              title={`${c.doc_title} — ${c.title}`}
              className="w-full text-left px-2.5 py-1.5 rounded-md hover:bg-gray-100 transition-colors"
            >
              <div className="text-[13px] text-gray-800 truncate">{c.title}</div>
              <div className="text-[11px] text-gray-400 truncate">{c.doc_title}</div>
            </button>
          ))}
        </div>
      </div>

      {/* 下：文档库 */}
      <div className="flex-1 flex flex-col min-h-0">
        <div className="px-3 py-2 flex items-center justify-between">
          <span className="text-xs font-semibold text-gray-500 tracking-wide">文档库</span>
          <div className="flex gap-1">
            <button onClick={() => setCatalogFilter('all')} className="px-2 py-0.5 text-[11px] rounded border border-gray-300 hover:bg-gray-100">
              目录
            </button>
            <button onClick={() => setView('discovery')} className="px-2 py-0.5 text-[11px] rounded bg-blue-600 text-white hover:bg-blue-700">
              发现
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto px-2 pb-2 min-h-0">
          <div className="text-[11px] text-gray-400 px-2 pt-1 pb-0.5">未读（{unread.length}）</div>
          <div className="space-y-0.5">
            {unread.map((d) => <DocItem key={d.id} doc={d} />)}
            {unread.length === 0 && <div className="text-[12px] text-gray-300 px-2">—</div>}
          </div>
          <div className="text-[11px] text-gray-400 px-2 pt-3 pb-0.5">已读（{read.length}）</div>
          <div className="space-y-0.5">
            {read.map((d) => <DocItem key={d.id} doc={d} />)}
            {read.length === 0 && <div className="text-[12px] text-gray-300 px-2">—</div>}
          </div>
        </div>
      </div>
    </aside>
  )
}
