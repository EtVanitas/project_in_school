/** 中栏：视图切换（欢迎/目录/发现/PDF/笔记）+ 顶部操作栏。 */

import { useStore } from '../store'
import CatalogView from './CatalogView'
import DiscoveryView from './DiscoveryView'
import NotesView from './NotesView'
import PdfViewer from './PdfViewer'

function WelcomeView() {
  const { docs, setView, setCatalogFilter } = useStore()
  const unread = docs.filter((d) => d.status === 'unread').length
  const read = docs.length - unread

  return (
    <div className="h-full flex items-center justify-center">
      <div className="max-w-2xl w-full px-8">
        <h1 className="text-2xl font-bold text-gray-800">智能论文阅读助手</h1>
        <p className="text-sm text-gray-500 mt-2 leading-relaxed">
          发现热门论文、下载到本地未读区、在网页里阅读并随时向 AI 提问，读完后标记已读，对话一键整理成笔记。
        </p>
        <div className="grid grid-cols-3 gap-4 mt-8">
          <button
            onClick={() => setView('discovery')}
            className="rounded-xl border border-gray-200 bg-white p-5 text-left hover:border-blue-400 hover:shadow-md transition-all"
          >
            <div className="text-2xl">🔥</div>
            <div className="font-semibold text-gray-800 mt-2">寻找最新最火论文</div>
            <div className="text-xs text-gray-500 mt-1">HF Papers 当日热门 / arXiv 搜索</div>
          </button>
          <button
            onClick={() => setCatalogFilter('unread')}
            className="rounded-xl border border-gray-200 bg-white p-5 text-left hover:border-blue-400 hover:shadow-md transition-all"
          >
            <div className="text-2xl">📖</div>
            <div className="font-semibold text-gray-800 mt-2">阅读未读论文</div>
            <div className="text-xs text-gray-500 mt-1">共 {unread} 篇待阅读</div>
          </button>
          <button
            onClick={() => setCatalogFilter('read')}
            className="rounded-xl border border-gray-200 bg-white p-5 text-left hover:border-blue-400 hover:shadow-md transition-all"
          >
            <div className="text-2xl">🗂️</div>
            <div className="font-semibold text-gray-800 mt-2">重读过去论文</div>
            <div className="text-xs text-gray-500 mt-1">已读 {read} 篇，含历史笔记</div>
          </button>
        </div>
      </div>
    </div>
  )
}

export default function CenterViewer() {
  const { view, currentDoc, markRead, deleteDoc, setView } = useStore()
  // 仅在阅读相关视图（PDF/笔记）显示当前文档的标题与操作
  const showDocHeader = (view === 'pdf' || view === 'notes') && !!currentDoc

  const confirmDelete = async () => {
    if (!currentDoc) return
    if (window.confirm(`确定删除《${currentDoc.title}》？其对话与笔记将一并删除。`)) {
      await deleteDoc(currentDoc)
    }
  }

  return (
    <main className="flex-1 min-w-0 flex flex-col bg-gray-50">
      {/* 顶部操作栏 */}
      <header className="h-12 shrink-0 bg-white border-b border-gray-200 flex items-center gap-2 px-4 [&>button]:shrink-0 [&>button]:whitespace-nowrap">
        <button
          onClick={() => setView('welcome')}
          className={`text-sm px-2 py-1 rounded ${view === 'welcome' ? 'text-blue-700 font-medium' : 'text-gray-500 hover:bg-gray-100'}`}
        >
          首页
        </button>
        <button
          onClick={() => setView('catalog')}
          className={`text-sm px-2 py-1 rounded ${view === 'catalog' ? 'text-blue-700 font-medium' : 'text-gray-500 hover:bg-gray-100'}`}
        >
          目录
        </button>
        <button
          onClick={() => setView('discovery')}
          className={`text-sm px-2 py-1 rounded ${view === 'discovery' ? 'text-blue-700 font-medium' : 'text-gray-500 hover:bg-gray-100'}`}
        >
          发现
        </button>

        <div className="flex-1 min-w-0 mx-2 text-sm text-gray-700 truncate">
          {showDocHeader && currentDoc ? currentDoc.title : ''}
        </div>

        {showDocHeader && currentDoc && (
          <div className="flex items-center gap-1.5 shrink-0">
            <span className={`text-[11px] px-1.5 py-0.5 rounded ${currentDoc.status === 'read' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'}`}>
              {currentDoc.status === 'read' ? '已读' : '未读'}
            </span>
            <button
              onClick={() => setView('notes')}
              className="text-xs px-2 py-1 rounded border border-gray-300 hover:bg-gray-100"
            >
              笔记
            </button>
            <button
              onClick={() => markRead(currentDoc, currentDoc.status === 'unread')}
              className={`text-xs px-2 py-1 rounded ${currentDoc.status === 'unread' ? 'bg-green-600 text-white hover:bg-green-700' : 'border border-gray-300 hover:bg-gray-100'}`}
            >
              {currentDoc.status === 'unread' ? '标记已读' : '退回未读'}
            </button>
            <button onClick={confirmDelete} className="text-xs px-2 py-1 rounded border border-red-200 text-red-600 hover:bg-red-50">
              删除
            </button>
          </div>
        )}
      </header>

      {/* 内容区 */}
      <div className="flex-1 min-h-0">
        {view === 'welcome' && <WelcomeView />}
        {view === 'catalog' && <CatalogView />}
        {view === 'discovery' && <DiscoveryView />}
        {view === 'notes' && <NotesView />}
        {view === 'pdf' && (currentDoc ? <PdfViewer /> : <WelcomeView />)}
      </div>
    </main>
  )
}
