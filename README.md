# 🧠 AI Health Journal

An LLM-powered journaling assistant that helps you reflect, gain insights, and track your mental health — all locally and securely.

## 🚀 Features

- ✨ AI-generated journaling prompts to get started.
- 💡 Insight generation from your personal entries.
- 📜 Chat history with interactive summaries.
- 🌗 Light/Dark theme toggle for better accessibility.
- 🖥️ Fully responsive layout.
- 📂 Local-first architecture (no data leaves your machine).

## 🧰 Tech Stack

- **Frontend**: HTML, CSS, Vanilla JS
- **Backend**: Flask (Python)
- **LLM Integration**: Local LLM APIs (e.g., Phi-3, Mistral via Ollama)
- **Persistence**: JSON-based local session storage

## 📝 How It Works

1. Write a journal entry in natural language.
2. Click "Analyze" to send it to the local LLM API.
3. Receive an insight or reflection.
4. View past entries in the **History Panel** (toggleable).
5. Generate a fresh prompt if you're feeling stuck.
6. Start a new session anytime.

## 🖼️ UI Overview

- **Main Panel**: For writing and viewing current insights.
- **Sidebar**: Shows concise titles from past entries.
- **Theme & History** toggles in the corners.
- **Smooth animations** and transitions included.

## 🧪 Run Locally

```bash
git clone https://github.com/Chunduri-Aditya/ai-health-journal.git
cd ai-health-journal
python3 -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Visit: `http://127.0.0.1:5000/`

## 💬 Example Models

- `phi3:3.8b` for journaling insights
- `samantha-mistral:7b` for prompt generation

Ensure these models are served via your local API (e.g., `http://localhost:11434`).

## ✅ Future Roadmap

- 📈 Analytics dashboard
- 📦 Export sessions to PDF
- 🔐 Password-protected journal access
- 📊 Sentiment graphing over time

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first to discuss.

## 📄 License

[MIT License](LICENSE)

---

Made with ❤️ by [Aditya Chunduri](https://github.com/Chunduri-Aditya)
