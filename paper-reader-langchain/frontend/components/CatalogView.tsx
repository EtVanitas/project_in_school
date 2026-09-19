/** 目录视图：全部文档的摘要卡片（类似书籍目录），支持未读/已读筛选。 */

import { useStore } from '../store'
import type { Doc } from '../api'

function CatalogCard({ doc }: { doc: Doc }) {
  const { openDoc, markRead } = useStore()

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4 flex flex-col hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between gap-2">
        <button
          onClick={() => openDoc(doc)}
          className="text-left font-medium text-[15px] text-gray-900 hover:text-blue-700 leading-snug"
        >
          {doc.title}
        </button>
        <span className={`shrink-0 text-[11px] px-1.5 py-0.5 rounded ${doc.status === 'read' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
          {doc.status === 'read' ? '已读' : '未读'}
        </span>
      </div>
      <div className="text-xs text-gray-400 mt-1.5 flex items-center gap-2 flex-wrap">
        <span>{doc.source === 'legacy' ? '本地导入' : doc.source === 'hf_papers' ? 'HF 热门' : 'arXiv'}</span>
        {doc.arxiv_id && <span>{doc.arxiv_id}</span>}
        {doc.published && <span>{doc.published}</span>}
        {(doc.note_count ?? 0) > 0 && <span className="text-blue-600">笔记 {doc.note_count}</span>}
      </div>
      <div className="text-xs text-gray-500 mt-1 truncate">{doc.authors}</div>
      <p className="text-[13px] text-gray-600 mt-2 leading-relaxed line-clamp-4 flex-1" title={doc.abstract}>
        {doc.abstract || '（暂无摘要）'}
      </p>
      <div className="flex gap-2 mt-3">
        <button
          onClick={() => openDoc(doc)}
          className="text-xs px-3 py-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          阅读
        </button>
        <button
          onClick={() => markRead(doc, doc.status === 'unread')}
          className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg hover:bg-gray-100"
        >
          {doc.status === 'unread' ? '标记已读' : '退回未读'}
        </button>
      </div>
    </div>
  )
}

export default function CatalogView() {
  const { catalog, catalogFilter, setCatalogFilter } = useStore()
  const filtered = catalog.filter((d) => catalogFilter === 'all' || d.status === catalogFilter)

  const tab = (key: 'all' | 'unread' | 'read', label: string, count: number) => (
    <button
      onClick={() => setCatalogFilter(key)}
      className={`px-3 py-1.5 text-sm rounded-lg ${catalogFilter === key ? 'bg-blue-600 text-white' : 'bg-white border border-gray-300 text-gray-600 hover:bg-gray-50'}`}
    >
      {label}（{count}）
    </button>
  )

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-5xl mx-auto px-6 py-5">
        <div className="flex gap-2">
          {tab('all', '全部', catalog.length)}
          {tab('unread', '未读', catalog.filter((d) => d.status === 'unread').length)}
          {tab('read', '已读', catalog.filter((d) => d.status === 'read').length)}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5 mt-4">
          {filtered.map((d) => <CatalogCard key={d.id} doc={d} />)}
        </div>
        {filtered.length === 0 && (
          <div className="text-center text-gray-400 text-sm py-16">
            这里还没有论文 —— 去「发现」页抓取热门论文，或在左侧点击「发现」按钮
          </div>
        )}
      </div>
    </div>
  )
}
