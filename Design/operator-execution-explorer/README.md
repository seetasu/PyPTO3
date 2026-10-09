# PyPTO MatMul Explorer

Open `index.html` from a static server rooted at `PyPTO3/`.

The page follows the supplied reference image: header; top Source and Tensor panes; bottom Instructions pane with work-unit list, nested M/N loop groups, horizontal steps, and floating playback. The default visual state is `M1N3 / Slice B`.

The implementation consumes PTO `ide-frame`, `workbench-shell`, `matrix-canvas`, `floating-playback-control`, tokens, tabs, and buttons from `PyPTO3/vendor/pto-design-system`. It leaves the Ascend C Demo untouched. Its hardware-specific app script cannot be imported without introducing unsupported AIC and memory-hierarchy semantics.

The source pane is a compact excerpt of [upstream `examples/beginner/matmul.py`](https://github.com/hw-native-sys/pypto-lib/blob/main/examples/beginner/matmul.py); displayed line numbers are excerpt-local. Compiler and runtime evidence fields remain null.
