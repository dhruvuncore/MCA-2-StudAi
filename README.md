# AI Student Learning & Exam Assistant (RAG Pipeline)

A full-stack, AI-powered study platform featuring multimodal document processing (PDF, Word, Markdown, handwritten notes & scans), vector retrieval (RAG), AI study tutor chat, smart notes synthesis, and university practice exam generator.

---

## 📋 Prerequisites for Mac (macOS)

Before starting, ensure your Mac has:

1. **macOS**: Compatible with all modern macOS versions (Ventura, Sonoma, Sequoia, etc.) on both Apple Silicon (M1/M2/M3/M4) and Intel Macs.
2. **Node.js**: Version **18.x, 20.x, or 22.x LTS** (Node 20+ recommended).
   - Check if installed in Terminal:
     ```bash
     node -v
     npm -v
     ```
   - If not installed, install via [Homebrew](https://brew.sh/):
     ```bash
     brew install node
     ```
     *Or download the macOS installer directly from [nodejs.org](https://nodejs.org).*
3. **Terminal**: Built-in macOS Terminal, iTerm2, or Warp.

---

## 🔑 Step 1: Get Your Free Google Gemini API Key

This project uses the official `@google/genai` SDK with Google Gemini models (e.g., `gemini-3.8-flash` / `gemini-3.5-flash`) for AI tutoring, OCR handwriting transcription, smart notes generation, and exam generation.

1. Open your browser and go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Sign in with your Google Account.
3. Click **"Create API key"** (or **"Get API key"**).
4. Select or create a Google Cloud project, then click **"Create API key in existing project"**.
5. Copy your generated API key (it starts with `AIzaSy...`).

---

## 💻 Step 2: Set Up the Project on Your Mac

### 1. Open Terminal and Navigate to the Folder
If you extracted a `.zip` archive or cloned the repository, navigate into the project directory:
```bash
cd ~/Downloads/study-ai-rag-project
```
*(Replace with the actual path to your project folder).*

### 2. Configure Your `.env` File
Create your local environment file by copying `.env.example`:
```bash
cp .env.example .env
```

Open `.env` in your text editor (or use `nano .env` in Terminal) and replace `MY_GEMINI_API_KEY` with your actual key:
```bash
# .env
GEMINI_API_KEY="AIzaSyYourActualKeyGoesHere"
PORT=3000
```
> **Tip**: Even if you don't supply an API key initially, the application has an integrated local academic synthesis engine and deterministic semantic vector embedding model, allowing offline operation!

### 3. Install Dependencies
Run npm in your project root to install all required client and server packages:
```bash
npm install
```

---

## 🚀 Step 3: Run the Application

### Development Mode (Recommended)
To run the full-stack server (Node.js/Express backend + Vite React frontend on port 3000):

```bash
npm run dev
```

You will see:
```text
AI Student RAG Server running on http://0.0.0.0:3000
```

### Access in Your Browser
Open Safari, Chrome, Arc, or Firefox and go to:
👉 **[http://localhost:3000](http://localhost:3000)**

---

## 📦 Production Build & Run (Optional)

To test a production-ready compiled build:

1. **Build the client and bundle the server**:
   ```bash
   npm run build
   ```
2. **Start the production server**:
   ```bash
   npm run start
   ```

---

## 🛠️ Helpful Mac Tips & Troubleshooting

### Issue 1: Port 3000 is Already in Use
If another process (like a previous server or macOS service) is occupying port 3000:
1. Find what is using port 3000:
   ```bash
   lsof -i :3000
   ```
2. Either stop that process:
   ```bash
   kill -9 <PID>
   ```
3. Or launch this project on another port (e.g., 3001):
   ```bash
   PORT=3001 npm run dev
   ```
   Then open `http://localhost:3001`.

### Issue 2: Permission Errors During `npm install`
Ensure you do NOT use `sudo npm install`. If permissions were corrupted previously:
```bash
sudo chown -R $(whoami) ~/.npm
npm install
```

### Issue 3: Stored Data and Uploads Location
Uploaded documents, vector index files, and parsed chunks are stored locally under:
- `data/storage/` – Uploaded PDFs, docs, and images
- `data/db/` – Document metadata and text chunks
- `data/vectordb/` – Vector similarity embeddings

To reset your local database, simply delete the files inside `data/db/` and `data/vectordb/`.

---

## 📂 Project Architecture

- **`server.ts`**: Express backend entrypoint mounting Vite middleware in development and serving static assets in production.
- **`server/api/`**: REST endpoints for document uploading, OCR transcription, vector search, and LLM chat.
- **`server/services/`**: RAG pipeline (chunking, semantic embeddings, vector index, and Gemini LLM grounding).
- **`src/`**: Modern React 19 interface styled with Tailwind CSS v4, Lucide icons, and Recharts.
