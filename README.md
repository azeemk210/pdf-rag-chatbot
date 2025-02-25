# 📚 Local DeepSeek RAG

## 🚀 Installation & Setup Guide

This guide explains how to install, set up, and use the **Local DeepSeek RAG** system on your machine.

---

## 📌 **1. Prerequisites**
- Python 3.10+
- Git
- Conda (Optional but recommended)

---

## 🔧 **2. Clone the Repository**
```bash
# Clone the repo
git clone https://github.com/azeemk210/pdf-rag-chatbot.git
cd local_deepseek_rag
```

---

## 🛠 **3. Create and Activate Virtual Environment**
### **Using Conda** (Recommended)
```bash
conda create --name deepseek_rag python=3.10 -y
conda activate deepseek_rag
```

### **Using Virtualenv** (Alternative)
```bash
python -m venv deepseek_rag
source deepseek_rag/bin/activate  # On macOS/Linux
deepseek_rag\Scripts\activate    # On Windows
```

---

## 🔽 **4. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🏗 **5. Install & Set Up Ollama**

### **Step 1: Download & Install Ollama**
- **Windows/macOS/Linux**: Download Ollama from the official site:
  - [🔗 Ollama Install](https://ollama.com/download)


### **Step 2: Download llama2 model**
```bash
ollama run llama2
```

### **Step 3: Start Ollama Server**
```bash
ollama serve
```


---

## ⚡ **6. Run the Application**
```bash
python localrag.py
```

- Open the **Gradio UI** in your browser:
  ```
  http://127.0.0.1:7860
  ```
- Upload PDFs and start asking questions!

---

## 📝 **7. Usage**
### **Uploading Documents**
- Click on "📂 Upload PDF Documents" and select your files.
- Click **Initialize System** to process the documents.

### **Asking Questions**
- Enter a question related to the document in the text box.
- Click **Ask 🤖** and view the response in the chat.

### **Downloading Conversation History**
- Click **💾 Download Chat History** to save your Q&A session.

---

## ❌ **8. Troubleshooting**
### **Ollama Not Connecting?**
```bash
ollama serve
```
Ensure it's running and accessible at `http://127.0.0.1:11434` or update:
```python
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11436"
```

### **Wrong Document Answers?**
- Delete old embeddings:
```bash
rm -r chroma_db
```
- Restart the system and re-upload PDFs.

---

## 🤝 **Contributing**
Feel free to fork this repository and submit pull requests!


