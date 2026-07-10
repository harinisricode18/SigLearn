# 🫀 SigLearn

> An AI-powered biomedical signal analysis and learning platform for students, researchers, and educators.

SigLearn is an interactive web application that allows users to explore and understand biomedical signals through signal processing, visualization, machine learning, and an AI tutor.

Rather than focusing only on classification, SigLearn aims to help users **learn why a signal looks the way it does** by combining traditional signal analysis with conversational AI.

---

## ✨ Features

- 📈 Interactive visualization of biomedical signals
- 🧠 AI Tutor for concept-based explanations
- ⚡ Automatic preprocessing pipeline
- 📊 Signal parameter extraction
- 🤖 Machine Learning signal classification
- 📝 Processing log for transparency
- 💬 Context-aware AI chat based on the uploaded signal

---

## Supported Signals

| Signal | Dataset |
|---------|---------|
| ❤️ ECG | MIT-BIH Arrhythmia Database |
| 🧠 EEG | EEGBCI (PhysioNet via MNE) |
| 💪 EMG | Ninapro DB1 |
| 🫀 PPG | BIDMC PPG & Respiration Database |
| 🫁 Respiration | BIDMC PPG & Respiration Database |

---

## AI Tutor

SigLearn includes an integrated AI tutor that can answer questions about the currently loaded signal.

Examples include:

- Explain this signal.
- What does Alpha power mean?
- Why is this classified as EEG?
- Is this signal noisy?
- What preprocessing was applied?
- What is the clinical significance?

The tutor uses the extracted parameters and signal metadata to provide context-aware educational explanations while avoiding unsupported medical diagnosis.

---

## Processing Pipeline

The preprocessing pipeline currently includes:

- Signal loading
- Normalization
- Bandpass filtering
- Noise reduction
- Peak/event detection
- Feature extraction
- Machine learning classification
- AI-assisted explanation

---

## Project Structure

```
SigLearn/
│
├── ai/
│   └── explainer.py
│
├── core/
│   ├── loaders.py
│   ├── preprocessor.py
│   ├── feature_extractor.py
│   ├── classifier.py
│   └── peak_detector.py
│
├── src/
│   └── streamlit_app.py
│
├── ui/
│   └── plot_panel.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Datasets

Large biomedical datasets are hosted separately using a Hugging Face Dataset repository to keep the application lightweight.

Datasets used include:

- MIT-BIH Arrhythmia Database
- EEGBCI Dataset
- Ninapro DB1
- BIDMC PPG & Respiration Database

---

## Technologies

- Python
- Streamlit
- NumPy
- SciPy
- Pandas
- MNE
- WFDB
- scikit-learn
- Hugging Face Hub
- OpenAI-compatible Hugging Face Router API

---

## Installation

Clone the repository

```bash
git clone https://github.com/your_username/SigLearn.git
cd SigLearn
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
HF_TOKEN=your_huggingface_token
```

Run the application

```bash
streamlit run app.py
```

---

## Future Improvements

- More advanced ML models
- Deep learning classifiers
- Explainable AI visualizations
- Additional biomedical signal support
- Signal quality assessment
- Report generation
- User-uploaded dataset management

---

## Disclaimer

SigLearn is intended for **educational and research purposes only**.

The application does **not** provide medical diagnosis or clinical decision support.

---

## Author

**Harinisri Ramesh**

Biomedical Engineering | AI & Machine Learning | Biomedical Signal Processing

---

## License

This project is licensed under the MIT License.
