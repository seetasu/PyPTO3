# Callable Explorer · csa_merge_pack_publish

本副本已接入新 callable，正式 `../csa-pass-field` 未修改。

预览：`http://127.0.0.1:4217/`，默认显示 P35；地址视图可打开 `#pass=36&palette=identity&layout=horizontal&view=compile`。

- 52 个前端/Pass 状态，目标 scope→InCore→单 AIV。
- 默认主图显示当前阶段实际算子类型及次数、循环形式与边界，按六个显式配置区域组织；双击展开逐语句算子，详情保留值、表达式及来源。
- 各阶段真实 Vec 分配与 MemRef 绑定；P33 为 53 个，P35 后为 12 个。选择区域/展开算子/分配可高亮 UB 对应范围。
- 最终 ISA：68 处 Tile + 3 处通信静态出现位置；按原始源码区域映射并按 Op 区域对齐。
- 保留已确认图元、选中与悬停、Pass 导航、无点阵背景、折叠 Inspector。

命令：`rtk npm run build`、`rtk npm run test:callable`、`rtk npm run dev`。

数据入口：`data/callable/`。`scripts/probe-callable.py` 提取静态语法，`scripts/build-callable.py` 使用本样本的显式语义分区和 loc 配置生成数据；`src/CallableField.jsx` 负责画布适配。不是任意 callable 的全自动语义划分，也尚未提供上传入口。展开算子的数据边表达静态语句读写，完整 CFG/循环回边仍待扩展。

已完成构建、数据与几何验证；浏览器访问被当前策略阻止，尚未完成视觉烟测。

## 通用化规则与验证

- [Callable Explorer 通用化规则](docs/callable-explorer-rules.md)
- [其他 callable 的验证结论与复现方式](docs/cross-callable-validation.md)
- [364 个阶段的原始审计结果](docs/validation/cross-callable-audit.json)

当前仍是样本适配，不代表任意 callable 上传已实现；验证文档区分源码审计、UI 接入和视觉验收。

## Callable selector

底部 Pass 导航左侧可切换 `qk_pv` 和 `csa_merge_pack_publish`，URL 用 `callable` 保存样本身份。切换保留 Pass/ISA、布局及染色设置，清除旧节点选择并重新加载样本，播放和手动位置随之重置。qk_pv 暂时复用原样本适配器；这不是统一语义模型已完成的声明。其他五个审计样本尚未进入可用列表。

运行 `npm run test:samples` 检查两样本注册、切换 URL、阶段投影和内存空间隔离。
