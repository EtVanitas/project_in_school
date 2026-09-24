/** 笔记视图：当前文档的笔记查看/编辑 + 搜索功能（v4 简化版：单文件每篇论文）。 */

import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useStore } from '../store'
import { api } from '../api'
import type { Note } from '../api'

function NoteEditor({ arxivId, title, initialContent }: { arxivId: string; title: string; initialContent: string }) {
  const [content, setContent] = useState(initialContent)
  const [saving, setSaving] = useState(false)

  const handleSave = async () => {
    setSaving(true)
    try {
      await api.updateNote(arxivId, content)
      alert('笔记已保存')
    } catch (e) {
      console.error('保存失败:', e)
      alert('保存失败，请重试')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs text-gray-400">{title} ({arxivId})</div>
        <button
          onClick={handleSave}
          disabled={saving}
          className="text-xs px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {saving ? '保存中…' : '保存'}
        </button>
      </div>
      <textarea
        value={content}
        onChange={(e) => setContent(e.target.value)}
        className="w-full h-96 text-[13px] leading-relaxed border border-gray-300 rounded-lg p-3 focus:outline-none focus:border-blue-500 font-mono"
        placeholder="在此编辑笔记..."
      />
    </div>
  )
}



export default function NotesView() {
  const { currentDoc, setView } = useStore()
  const [activeNote, setActiveNote] = useState<string | null>(null)
  const [noteContent, setNoteContent] = useState('')
  const [loading, setLoading] = useState(false)

  // 加载现有笔记内容
  useEffect(() => {
    if (currentDoc && activeNote === currentDoc.arxiv_id) {
      setLoading(true)
      api.getNote(currentDoc.arxiv_id)
        .then(note => {
          setNoteContent(note.content || '')
        })
        .catch(err => {
          console.error('加载笔记失败:', err)
          setNoteContent('')
        })
        .finally(() => setLoading(false))
    }
  }, [currentDoc, activeNote])

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-800">笔记</h2>
          {currentDoc && (
            <button 
              onClick={() => setView('pdf')} 
              className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg hover:bg-gray-100"
            >
              返回阅读
            </button>
          )}
        </div>
        <p className="text-xs text-gray-400 mt-0.5">
          {currentDoc ? `当前文档：${currentDoc.title}` : '所有笔记'}
        </p>
        {/* 显示当前论文的笔记 */}
        {currentDoc && activeNote === currentDoc.arxiv_id ? (
          loading ? (
            <div className="text-center text-gray-400 py-16">加载中…</div>
          ) : (
            <NoteEditor 
              arxivId={currentDoc.arxiv_id} 
              title={currentDoc.title} 
              initialContent={noteContent}
            />
          )
        ) : (
          <div className="text-center text-gray-400 text-sm py-16">
            {currentDoc ? (
              <>
                <p className="mb-3">点击「查看笔记」开始为该论文添加笔记</p>
                <button 
                  onClick={() => setActiveNote(currentDoc.arxiv_id)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
                >
                  查看笔记
                </button>
              </>
            ) : (
              '请先选择一篇论文'
            )}
          </div>
        )}
      </div>
    </div>
  )
}
