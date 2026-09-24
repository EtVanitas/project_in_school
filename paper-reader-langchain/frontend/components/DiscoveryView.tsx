/** 发现视图：HF Papers 日/周/月榜单（个性化重排 + 抓取兜底提示）/ arXiv 搜索，勾选确认下载。 */

import { useState } from 'react'
import { api, Candidate } from '../api'
import { useStore } from '../store'

type Source = 'hf_papers' | 'arxiv'
type Period = 'daily' | 'weekly' | 'monthly'

const PERIOD_LABEL: Record<Period, string> = { daily: '当日', weekly: '本周', monthly: '本月' }

export default function DiscoveryView() {
  const { refreshDocs, refreshCatalog } = useStore()
  const [source, setSource] = useState<Source>('hf_papers')
  const [period, setPeriod] = useState<Period>('daily')
  const [query, setQuery] = useState('')
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [expanded, setExpanded] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const [message, setMessage] = useState('')
  const [staleAt, setStaleAt] = useState('')  // 非空 = 实时抓取失败，展示的是该时刻的缓存
  const [directId, setDirectId] = useState('')

  const fetchCandidates = async (src: Source, p: Period = period) => {
    setLoading(true)
    setMessage('')
    setCandidates([])
    setSelected(new Set())
    setStaleAt('')
    try {
      const r = await api.discover(src, src === 'arxiv' ? query.trim() : '', 20, p)
      setCandidates(r.items)
      if (r.stale) setStaleAt(r.cached_at)
    } catch (e) {
      setMessage(`获取候选失败：${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  const toggle = (id: string) => {
    const next = new Set(selected)
    if (next.has(id)) next.delete(id)
    else next.add(id)
    setSelected(next)
  }

  const downloadSelected = async () => {
    const items = candidates.filter((c) => selected.has(c.arxiv_id))
    if (items.length === 0) return
    setDownloading(true)
    setMessage('')
    try {
      const r = await api.download({ items })
      const parts = [
        ...r.downloaded.map((m) => `✓ ${m}`),
        ...r.skipped.map((m) => `• ${m}`),
        ...r.errors.map((m) => `✗ ${m}`),
      ]
      setMessage(parts.join('\n'))
      setSelected(new Set())
      await refreshDocs()
      await refreshCatalog()
    } catch (e) {
      setMessage(`下载失败：${(e as Error).message}`)
    } finally {
      setDownloading(false)
    }
  }

  const downloadById = async () => {
    if (!directId.trim()) return
    setDownloading(true)
    setMessage('')
    try {
      const r = await api.download({ arxiv_id: directId.trim() })
      setMessage([...r.downloaded, ...r.skipped, ...r.errors.map((m) => `✗ ${m}`)].join('\n') || '完成')
      setDirectId('')
      await refreshDocs()
      await refreshCatalog()
    } catch (e) {
      setMessage(`下载失败：${(e as Error).message}`)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 py-5">
        {/* 数据源切换 */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            <button
              onClick={() => { setSource('hf_papers'); fetchCandidates('hf_papers') }}
              className={`px-3 py-1.5 text-sm ${source === 'hf_papers' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
            >
              HF 热门
            </button>
            <button
              onClick={() => setSource('arxiv')}
              className={`px-3 py-1.5 text-sm ${source === 'arxiv' ? 'bg-blue-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
            >
              arXiv 搜索
            </button>
          </div>
          {source === 'hf_papers' && (
            <>
              <div className="flex rounded-lg border border-gray-300 overflow-hidden">
                {(Object.keys(PERIOD_LABEL) as Period[]).map((p) => (
                  <button
                    key={p}
                    onClick={() => { setPeriod(p); fetchCandidates('hf_papers', p) }}
                    disabled={loading}
                    className={`px-3 py-1.5 text-sm disabled:opacity-50 ${period === p ? 'bg-violet-600 text-white' : 'bg-white text-gray-600 hover:bg-gray-50'}`}
                  >
                    {PERIOD_LABEL[p]}
                  </button>
                ))}
              </div>
              <button
                onClick={() => fetchCandidates('hf_papers')}
                disabled={loading}
                className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? '抓取中…' : '刷新'}
              </button>
            </>
          )}
          {source === 'arxiv' && (
            <>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && fetchCandidates('arxiv')}
                placeholder="关键词（建议英文，如 diffusion model）"
                className="flex-1 min-w-52 px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
              />
              <button
                onClick={() => fetchCandidates('arxiv')}
                disabled={loading}
                className="px-4 py-1.5 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                {loading ? '搜索中…' : '搜索'}
              </button>
            </>
          )}
        </div>

        {/* 直接按 ID 下载 */}
        <div className="flex items-center gap-2 mt-3">
          <input
            value={directId}
            onChange={(e) => setDirectId(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && downloadById()}
            placeholder="已知 arXiv ID 直接下载，如 2509.06942"
            className="w-72 px-3 py-1.5 text-sm border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
          />
          <button
            onClick={downloadById}
            disabled={downloading || !directId.trim()}
            className="px-4 py-1.5 text-sm border border-blue-600 text-blue-600 rounded-lg hover:bg-blue-50 disabled:opacity-50"
          >
            下载
          </button>
        </div>

        {/* 抓取兜底提示 */}
        {staleAt && (
          <div className="mt-3 text-xs bg-amber-50 border border-amber-300 text-amber-800 rounded-lg px-3 py-2">
            实时抓取失败，当前展示本地缓存榜单（{staleAt} 抓取）；恢复网络后点「刷新」重试
          </div>
        )}

        {/* 结果提示 */}
        {message && (
          <pre className="mt-3 text-xs whitespace-pre-wrap bg-white border border-gray-200 rounded-lg p-3 text-gray-700">{message}</pre>
        )}

        {/* 操作条 */}
        {candidates.length > 0 && (
          <div className="flex items-center justify-between mt-4 mb-2">
            <div className="text-sm text-gray-600">
              共 {candidates.length} 篇候选，已选 {selected.size} 篇
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setSelected(selected.size === candidates.length ? new Set() : new Set(candidates.map((c) => c.arxiv_id)))}
                className="text-xs px-3 py-1.5 border border-gray-300 rounded-lg hover:bg-gray-100"
              >
                {selected.size === candidates.length ? '取消全选' : '全选'}
              </button>
              <button
                onClick={downloadSelected}
                disabled={selected.size === 0 || downloading}
                className="text-xs px-4 py-1.5 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
              >
                {downloading ? '下载中…' : `确认下载（${selected.size}）`}
              </button>
            </div>
          </div>
        )}

        {/* 候选列表 */}
        <div className="space-y-2.5 mt-3">
          {candidates.map((c) => (
            <div key={c.arxiv_id} className="bg-white border border-gray-200 rounded-xl p-4 hover:border-gray-300 transition-colors">
              <div className="flex gap-3">
                <input
                  type="checkbox"
                  checked={selected.has(c.arxiv_id)}
                  onChange={() => toggle(c.arxiv_id)}
                  className="mt-1 w-4 h-4 accent-blue-600 shrink-0"
                />
                <div className="min-w-0 flex-1">
                  <div className="flex items-start gap-2">
                    <a
                      href={c.url}
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium text-[15px] text-gray-900 hover:text-blue-700 leading-snug"
                    >
                      {c.title}
                    </a>
                  </div>
                  <div className="text-xs text-gray-400 mt-1 flex items-center gap-2 flex-wrap">
                    <span>{c.arxiv_id}</span>
                    {c.published && <span>{c.published}</span>}
                    {c.source === 'hf_papers' && <span className="text-orange-600 font-medium">▲ {c.upvotes} upvotes</span>}
                    {c.reco && c.reco.length > 0 && (
                      <span className="text-violet-700 bg-violet-50 border border-violet-200 rounded-full px-2 py-0.5">
                        ★ 为你推荐：与你常读的 {c.reco.join(' / ')} 相关
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-1 truncate">{c.authors}</div>
                  <p
                    onClick={() => setExpanded(expanded === c.arxiv_id ? '' : c.arxiv_id)}
                    className={`text-[13px] text-gray-600 mt-2 leading-relaxed cursor-pointer ${expanded === c.arxiv_id ? '' : 'line-clamp-3'}`}
                    title="点击展开/收起摘要"
                  >
                    {c.abstract}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {candidates.length === 0 && !loading && (
          <div className="text-center text-gray-400 text-sm py-16">
            点击上方按钮抓取候选论文；勾选后「确认下载」存入未读区
          </div>
        )}
      </div>
    </div>
  )
}
