# Prototype Instructions

Run the local server yourself and open the preview in the browser available to this environment. Do not give the user server-start instructions when you can run it.

Before making substantial visual changes, use the Product Design plugin's `get-context` skill when the visual source is unclear or no longer matches the current goal. When the user gives durable prototype-specific design feedback, preferences, or decisions, record them in `AGENTS.md`.

When implementing from a selected generated mock, treat that image as the source of truth for layout, component anatomy, density, spacing, color, typography, visible content, and hierarchy.

## Durable design direction

- The default run overview should prioritize the current two-rank workload: give each rank its own compact 2.5D region and show observed cross-rank data relationships between regions.
- Keep multi-rank growth available through rank groups and focused expansion; do not flatten the default view into a table or fill it with generic process art.
- Rank regions should expose source-backed scheduler phases, task invocations, kernel names, declared dependencies, device trace observations, and generated artifacts. Mark unavailable fields explicitly and do not infer cross-rank causality or synchronized timing without source evidence.
- The run overview must help locate performance divergence, not just summarize counts: foreground same-Task-ID local duration comparisons, trace-ordered task context, communication stages, and per-task dependency counts. Keep Rank-local start offsets separate; compare measured durations without claiming aligned cross-rank starts.

Build app UI in `src/`. Keep `.openai/hosting.json`, `worker/index.js`, `scripts/prepare-sites-build.mjs`, and `tests/sites-worker.test.mjs` intact so the same local prototype can be handed to Sites. Before a Sites handoff, run `npm run build` and `npm run test:sites`; the build must leave `dist/client/index.html`, `dist/server/index.js`, and `dist/.openai/hosting.json`.
