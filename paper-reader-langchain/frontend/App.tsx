/** 三栏布局骨架：左（对话+文档库）/ 中（阅读与浏览）/ 右（问答对话）。 */

import { useEffect } from 'react'
import LeftPanel from './components/LeftPanel'
import CenterViewer from './components/CenterViewer'
import RightChat from './components/RightChat'
import { useStore } from './store'

export default function App() {
  const { refreshDocs, refreshCatalog, refreshConversations } = useStore()

  useEffect(() => {
    refreshDocs()
    refreshCatalog()
    refreshConversations()
  }, [refreshDocs, refreshCatalog, refreshConversations])

  return (
    <div className="h-screen flex bg-gray-100 text-gray-900 overflow-hidden min-w-[1180px]">
      <LeftPanel />
      <CenterViewer />
      <RightChat />
    </div>
  )
}
