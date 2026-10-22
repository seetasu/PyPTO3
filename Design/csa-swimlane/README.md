# CSA Execution

Open index.html; no build or network dependency. Derived by export.py from the local PyPTO3-main CSA run `_jit_l3_decode_csa_20260903_010617`. Source filenames and SHA-256 hashes are embedded per rank in data.js.

Only Worker View AIC/AIV execution events are included. Scheduler View duplicates are excluded. Each rank has 72 lanes and 4038 execution records. Flow endpoints are joined using bind_id to the original event id, scoped to Worker View. Duplicate event pairs are collapsed. Each rank has 142 mapped pairs; 172 flow pairs with endpoints outside the included core events are omitted. No inferred block-to-block edges or cross-rank alignment is added. These are execution windows, not a derived critical path.

Interactions: hover previews immediate dependencies; click pins task details; drag pans horizontally; Ctrl/Command-wheel zooms; Shift-wheel pans; ordinary scrolling moves through cores. Bottom navigator controls the visible time window independently of task selection. Rank selection uses independent trace origins.

Validation: node verify.cjs and node --check view.js. Browser smoke test was blocked by the browser file URL policy; visual verification remains pending. This page is not yet registered in the library or published.

UI revision: adaptive time ticks, compact all-core fit, Rows expansion, sparse core numbering, zoom-dependent clipped labels, draggable navigator with resize handles. Worker flow inspection found 4038 endpoints whose bind IDs are outside Worker X records; cross-view scheduler flows are not core dependency edges. 142 direct mapped pairs remain, not the full logical dependency graph.

## Color and diagnostic modes

Default Diagnostic mode compares duration within the same rank, exact logical task name, and core family (AIC/AIV). Flags require at least four peers and an absolute excess of at least 5 µs over the median; ratios 1.5/2/3 select yellow/orange/red. These are exploratory thresholds, not fault labels or causal findings. The inspector displays peer bars and the median. No cross-rank timing comparison is implied.

Task type is name-derived (wait/sync before communication before transfer before compute; unmatched names stay Other). Palette values are reused from `PyPTO3-main/vendor/pto-design-system/tokens/foundation.css`: l0b-deep-violet-300, copy-blue-300, l0a-violet-300, ark-domain-aux; diagnostic colors use mte-amber-300, ark-orange-500, ark-red-500. Local constants keep the standalone page portable.

Dependency links: long dashes are trace flows; dotted links come from fanout-hint logical dependencies, using a nearby core instance as a representative endpoint. They do not establish per-core synchronization.

Semantic correction: the initial broad Compute/Transfer palette is superseded by PTO model-graphviz SEMANTIC_COLOR_DEFAULTS (norm, linear, attention, rope, gate, comm). Runtime cache/layout and unknown/fused operations remain neutral. Name-derived mapping uses a primary semantic for fused kernels; communication waits keep collective semantics.
