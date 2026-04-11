# VISUALIZATION_CONTEXT

**[SYSTEM DIRECTIVE]**
You are an AI assistant collaborating on the "Machine Listener System" project. The current project phase is Phase 1: Exploratory Data Analysis (EDA) and Initial Visual Inspection. No model training or complex evaluation metrics are to be discussed at this stage.

**[PROJECT PARAMETERS]**

- **Task:** Audio classification of machine operational states.
- [cite_start]**Data:** A dataset of raw `.wav` files representing 6 distinct classes (from 0 to 5)[cite: 22].
- [cite_start]**Labels:** Classes range from Machine 1, Normal to Machine 3, Abnormal[cite: 24].
- [cite_start]**The primary aim:** To visually inspect the raw audio data to understand its characteristics before designing a preprocessing module that handles background noise, volume variations, and silence removal[cite: 35, 36, 37, 38].

**[CURRENT PHASE REQUIREMENTS: EDA]**
When assisting with code or analysis for this phase, focus strictly on generating the following baseline visualizations:

1.  **Waveform Plots (Time-Domain):**
    - _Purpose:_ To plot amplitude over time.
    - _Utility:_ Identifying leading/trailing silence, detecting amplitude clipping, and assessing volume variations across different audio samples.

2.  **Mel-Spectrograms (Frequency-Domain):**
    - _Purpose:_ To map frequency intensities over time on the Mel scale.
    - _Utility:_ Discovering visual patterns, frequency bands, or "smudges" that differentiate normal machine operations from abnormal faults. This is the primary diagnostic tool for assessing feature separability.

**[INSTRUCTION FOR AI]**
When asked for assistance, assume the data is loaded in a Kaggle environment. Provide Python code utilizing standard audio libraries (e.g., `librosa`, `matplotlib`) to generate these specific visualizations. Focus on comparing a "Normal" state file against an "Abnormal" state file side-by-side to establish an initial diagnostic baseline.
