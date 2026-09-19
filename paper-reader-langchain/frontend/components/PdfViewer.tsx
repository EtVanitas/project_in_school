/** PDF 阅读器：react-pdf 分页浏览 + 文本选择（选中内容上抛为引用条）+ 多模态解读本页。 */

import { useEffect, useMemo, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/TextLayer.css'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import { useStore } from '../store'

pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString()

export default function PdfViewer() {
  const { currentDoc, setQuote, interpretPage, streaming, streamKind, setView } = useStore()
  const [numPages, setNumPages] = useState(0)
  const [page, setPage] = useState(1)
  const [scale, setScale] = useState(1.25)
  const [loadError, setLoadError] = useState('')

  const file = useMemo(
    () => (currentDoc ? { url: `/api/docs/${currentDoc.id}/file` } : null),
    [currentDoc],
  )

  // 切换文档时重置页码
  const docId = currentDoc?.id
  useEffect(() => { setPage(1); setNumPages(0); setLoadError('') }, [docId])

  if (!currentDoc || !file) return null

  const handleMouseUp = () => {
    const text = window.getSelection()?.toString().trim() ?? ''
    if (text.length >= 4) setQuote(text)
  }

  return (
    <div className="h-full flex flex-col">
      {/* 工具栏 */}
      <div className="h-10 shrink-0 bg-white border-b border-gray-200 flex items-center gap-3 px-3 text-sm overflow-x-auto whitespace-nowrap [&>button]:shrink-0 [&>span]:shrink-0">
        <button
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page <= 1}
          className="px-2 py-0.5 rounded border border-gray-300 hover:bg-gray-100 disabled:opacity-40"
        >
          ‹ 上一页
        </button>
        <span className="text-gray-600 text-xs">
          {page} / {numPages || '…'}
        </span>
        <button
          onClick={() => setPage((p) => Math.min(numPages || p, p + 1))}
          disabled={numPages > 0 && page >= numPages}
          className="px-2 py-0.5 rounded border border-gray-300 hover:bg-gray-100 disabled:opacity-40"
        >
          下一页 ›
        </button>
        <span className="mx-2 text-gray-300">|</span>
        <button onClick={() => setScale((s) => Math.max(0.6, s - 0.15))} className="px-2 py-0.5 rounded border border-gray-300 hover:bg-gray-100">−</button>
        <span className="text-xs text-gray-500">{Math.round(scale * 100)}%</span>
        <button onClick={() => setScale((s) => Math.min(3, s + 0.15))} className="px-2 py-0.5 rounded border border-gray-300 hover:bg-gray-100">＋</button>
        <span className="mx-2 text-gray-300">|</span>
        <button
          onClick={() => interpretPage(page)}
          disabled={streaming || numPages === 0}
          className="px-2 py-0.5 rounded border border-violet-300 text-violet-700 hover:bg-violet-50 disabled:opacity-40"
          title="用多模态模型解读本页的图/表/公式（结果出现在右侧对话）"
        >
          {streaming && streamKind === 'vision' ? '解读中…' : '🖼️ 解读本页'}
        </button>
        <span className="mx-2 text-gray-300">|</span>
        <button
          onClick={() => setView('notes')}
          className="px-2 py-0.5 rounded border border-gray-300 hover:bg-gray-100"
        >
          📝 笔记
        </button>
        <span className="ml-4 text-[11px] text-gray-400">选中文本即可在右侧引用提问</span>
      </div>

      {/* 页面区 */}
      <div className="flex-1 overflow-auto flex justify-center py-4" onMouseUp={handleMouseUp}>
        {loadError ? (
          <div className="text-sm text-red-500 mt-16">{loadError}</div>
        ) : (
          <Document
            file={file}
            onLoadSuccess={({ numPages: n }) => setNumPages(n)}
            onLoadError={(e) => setLoadError(`PDF 加载失败：${e.message}`)}
            loading={<div className="text-sm text-gray-400 mt-16">PDF 加载中…</div>}
            error={<div className="text-sm text-red-500 mt-16">PDF 加载失败</div>}
          >
            <Page
              pageNumber={page}
              scale={scale}
              renderTextLayer
              renderAnnotationLayer
              loading={<div className="text-sm text-gray-400 mt-16">页面渲染中…</div>}
            />
          </Document>
        )}
      </div>
    </div>
  )
}
