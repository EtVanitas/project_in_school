/** 笔记勘误卡片：Agent 核实「笔记与论文矛盾」后提交的建议，用户确认后更正笔记。 */

import { useState } from 'react'
import { useStore } from '../store'
import type { NoteIssue } from '../api'

export default function NoteIssueCard({ issue }: { issue: NoteIssue }) {
  const { applyNoteIssue, dismissNoteIssue } = useStore()
  const [applying, setApplying] = useState(false)

  const apply = async () => {
    setApplying(true)
    try {
      await applyNoteIssue(issue.id)
    } finally {
      setApplying(false)
    }
  }

  return (
    <div className="border border-rose-200 bg-rose-50/70 rounded-xl px-3 py-2.5 space-y-1.5">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-semibold text-rose-700">笔记勘误建议</span>
        {issue.evidence_page > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-rose-100 text-rose-600">
            依据：第 {issue.evidence_page} 页
          </span>
        )}
      </div>
      <div className="text-[11px] leading-snug text-gray-700">
        <span className="text-gray-400">原句：</span>
        <span className="line-through decoration-rose-400/70">{issue.original_text}</span>
      </div>
      <div className="text-[11px] leading-snug text-gray-700">
        <span className="text-gray-400">修正：</span>{issue.correction}
      </div>
      {issue.reason && <div className="text-[11px] leading-snug text-gray-500">说明：{issue.reason}</div>}
      {issue.error && <div className="text-[11px] text-rose-600">更正失败：{issue.error}</div>}
      {issue.status === 'applied' ? (
        <div className="text-[11px] text-emerald-600">✓ 已更正笔记，下轮问答生效</div>
      ) : (
        <div className="flex items-center gap-2 pt-0.5">
          <button
            onClick={apply}
            disabled={applying}
            className="text-[11px] px-2.5 py-1 bg-rose-600 text-white rounded hover:bg-rose-700 disabled:opacity-50"
          >
            {applying ? '更正中…' : '更正笔记'}
          </button>
          <button
            onClick={() => dismissNoteIssue(issue.id)}
            className="text-[11px] px-2.5 py-1 border border-gray-300 rounded text-gray-500 hover:bg-gray-100"
          >
            忽略
          </button>
        </div>
      )}
    </div>
  )
}
