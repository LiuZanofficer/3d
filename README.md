# 3D Scene Understanding Research Workspace

Open-vocabulary 3D scene segmentation research built on
[NVlabs/Mosaic3D](https://github.com/NVlabs/Mosaic3D) (CVPR 2025).

## 📁 Repository Structure

| Path | What it is |
|------|------------|
| `Mosaic3D/` | Working copy of NVlabs/Mosaic3D **plus my research overlay** (see below) |
| `Mosaic3D/error_analysis/` | 🔬 **My research**: error-analysis & diagnosis reports, hypothesis tests, learned text/visual anchors, patches |
| `Mosaic3D/scripts/` | Upstream scripts + my `analyze_*` / `build_*` / `c1_*` analysis scripts |
| `Mosaic3D/logs/` | Training & eval logs |
| `lz/` | **LongTail-Zero3D (LZ)** — my own framework targeting long-tail classes, small objects & zero-shot transfer → [`lz/README.md`](lz/README.md) |
| `lz/artifacts/` | Categorized experiment artifacts (tmp scripts, outputs, analysis outputs, bundles) |
| `lz/docs/` | Diagrams & notes |

## 🔬 What's mine vs upstream

- **Upstream (Apache-2.0)**: `Mosaic3D/` codebase, configs, docker & CI scaffolding — see the
  [upstream repo](https://github.com/NVlabs/Mosaic3D) and its README (restored here for attribution).
- **My contributions**:
  - `Mosaic3D/error_analysis/` — the full diagnosis line of work
  - analysis scripts in `Mosaic3D/scripts/`
  - patches to `Mosaic3D/src/` (language module, eval, readout decorrelate)
  - the entire `lz/` project

## 🚀 Quick Start (LZ on a single RTX 4090)

Step-by-step guide: [`lz/RUN_STEPS_4090.txt`](lz/RUN_STEPS_4090.txt)
（中文版：[`lz/RUN_STEPS_4090_CN.txt`](lz/RUN_STEPS_4090_CN.txt)）

## 📌 Key findings so far

- `mask_text_vote` readout — the only validated gain so far (fg-mIoU **0.15475 → 0.17573**)
- Method B training + annotation-free readout diagnosis →
  [`Mosaic3D/error_analysis/reports/readout_failure_method_b.md`](Mosaic3D/error_analysis/reports/readout_failure_method_b.md)

## 🙏 Acknowledgements

- **Mosaic3D**: Lee, Junha et al., *Mosaic3D: Foundation Dataset and Model for Open-vocabulary 3D Segmentation*, CVPR 2025 ([NVlabs/Mosaic3D](https://github.com/NVlabs/Mosaic3D), Apache-2.0)

## 📄 License

Apache-2.0 (inherited from upstream) — see [LICENSE](LICENSE).
