# 外部 Demo 本地镜像：recommendation-trace-V3

下载日期：2026-10-29

## 来源

- 线上页面：<https://yinyucheng0601.github.io/compute-graph-viewer/operator-planning/recommendation-trace-V3.html>
- Demo 源码仓库：<https://github.com/yinyucheng0601/compute-graph-viewer>（main 分支，页面文件 Last-Modified 2026-10-22）
- 设计系统仓库：<https://github.com/yinyucheng0601/pto-design-system>（main 分支，该 demo 以 `../../pto-design-system/` 形式引用同一 Pages 站点下的另一个仓库）

## 目录结构

本目录按 GitHub Pages 站点根（`yinyucheng0601.github.io/`）的布局镜像，因此页面内的相对路径可直接在本地打开使用。

```
compute-graph-viewer/
  operator-planning/
    recommendation-trace-V3.html          # demo 主文件（单文件，471 KB，内联全部逻辑与数据）
    assets/planner-logo.png               # 页面引用
    assets/tail-strategy-*.svg            # 同目录配套插图（页面未直接引用，一并保留）
  components/
    candidate-population-lineage/         # 页面引用的局部组件（pattern.css / pattern.js / html / json / README）
pto-design-system/                        # 设计系统完整仓库（tokens、css/style.css、9 个被引用的 pattern 等）
```

## 依赖核对

页面引用的 24 个外部 CSS/JS 路径以及 `assets/planner-logo.png` 均已在本地齐备，已逐项核对。页面不含任何 `fetch` / XHR / 动态 `import`，所有数据内联，离线可运行。

## 已知缺口

- 页面底部三个候选方案链接指向 `recommendation-trace-V4.html`，该文件在上游仓库 `operator-planning/` 下尚不存在，属于上游的悬挂链接。
- 外链 `hiascend.com` 的 CANN 文档为站外引用，未下载。
