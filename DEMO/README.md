# DEMO Folder - Video Presentation Materials
## Team 15 - Hack the Horizon Hackathon

This folder contains all materials needed for the video presentation requested by the judges.

---

## 📁 Folder Contents

### Documentation Files

| File | Purpose | Duration Guide |
|------|---------|----------------|
| `VIDEO_SCRIPT.md` | Complete speaker script with timing | 8-10 minutes |
| `SLIDES_CONTENT.md` | 15 slides + 3 backup slides | Content for deck |
| `Q_AND_A_PREP.md` | 13 anticipated questions + answers | Judge Q&A prep |
| `DEMO_WALKTHROUGH.md` | Step-by-step dashboard demo guide | 2 min demo |
| `TECHNICAL_DEMO_SCRIPT.md` | **Code walkthrough with exact line numbers** | 6 min technical |
| `CODE_HIGHLIGHTS.md` | Key code snippets to show in video | Quick reference |

### Visual Assets (`figures/` folder)

**Generated Diagrams:**
- `qem_former_architecture_*.png` - Architecture flow diagram
- `qem_pipeline_overview_*.png` - Three-phase pipeline overview
- `results_summary_*.png` - Key metrics infographic
- `cdr_pauli_twirling_*.png` - CDR and Pauli Twirling explanation

**Benchmark Figures (copied from main project):**
- `architecture_evolution.png` - Model comparison chart
- `development_timeline.png` - Project progression
- `error_comparison.png` - Error bars across methods
- `improvement_ratio.png` - IR by circuit type
- `noise_model.png` - Noise model explanation
- `win_rate.png` - Win rate visualization

---

## 🎬 Video Recording Tips

### Equipment
- Microphone: Use external mic or headset for clear audio
- Screen: 1920x1080 resolution recommended
- Recording: OBS Studio, QuickTime, or Loom

### Presentation Flow
1. **Intro** (1 min) - Team introduction, problem statement
2. **Methodology** (3 min) - CDR, Pauli Twirling, QEM-Former
3. **Results** (2 min) - Benchmark metrics, honest failure analysis
4. **Demo** (2 min) - Live dashboard walkthrough
5. **Impact** (1 min) - Scalability, business value, conclusion

### Key Numbers to Emphasize
| Metric | Value |
|--------|-------|
| Variational Win Rate | **80%** |
| Error Reduction | **31.9%** |
| MSE Improvement | **3.3x** |
| Training Samples | 7,010 |
| QAOA Win Rate (failure) | 15% |

---

## 🚀 Quick Start

### Launch Dashboard for Demo
```bash
cd "/Users/abdulmalekbaitulmal/Downloads/Desktop/Quanta Related/AQC/AQC Hack The Horizon/QEM Codebase"
streamlit run dashboard.py
```

### Create Slides
Option 1: Copy content from `SLIDES_CONTENT.md` into PowerPoint/Google Slides
Option 2: Use the generated figures directly

---

## 📋 Pre-Recording Checklist

- [ ] Dashboard loads without errors
- [ ] All figures display correctly
- [ ] Microphone tested
- [ ] Screen recording software ready
- [ ] Script practiced at least once
- [ ] Timer visible during recording
- [ ] Quiet environment

---

## 👥 Team

- **Nakahosa Dinovic** - Resources Research, Reporter
- **Favour Idowu** - Validation Reviewer, Debugger
- **Abdulmalek Baitulmal** - Mentor, Solutions Integration

---

*Created for Hack the Horizon Hackathon - African Quantum Consortium*
