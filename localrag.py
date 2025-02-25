import os
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
import gradio as gr
from pypdf import PdfReader
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Set Ollama to use the correct port
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11436"

import shutil

def clear_embeddings():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Previous embeddings cleared.")


# Global variable for storing conversation history
conversation_history = []

# Initialize the RAG system
async def async_initialize_rag(pdf_files):
    try:
        print("Starting system initialization...")

        # Extract text from PDFs
        documents = []
        for pdf_file in pdf_files:
            with open(pdf_file.name, "rb") as f:
                pdf = PdfReader(f)
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        print(f"Extracted text from page {i+1}: {text[:200]}...\n")
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": pdf_file.name, "page": i+1}
                        ))
                    else:
                        print(f"Page {i+1} in {pdf_file.name} is empty.")

        if not documents:
            raise ValueError("No valid text found in uploaded PDFs.")

        # Split text into manageable chunks
        print("Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Reduce chunk size for faster processing
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Total chunks created: {len(chunks)}")

        # Generate embeddings for each chunk
        print("Generating embeddings...")
        embeddings = OllamaEmbeddings(model="llama2")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )

        # Create the QA pipeline
        print("Creating QA pipeline...")
        llm = OllamaLLM(model="llama2", temperature=0.3, num_ctx=2048)

        print("Initialization complete.")
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": min(3, len(chunks))}),
            return_source_documents=True
        )
    except Exception as e:
        print(f"Initialization Error: {str(e)}")
        raise RuntimeError(f"Initialization failed: {str(e)}")


# Process the user's question and generate an answer
async def async_ask_question(query, pipeline, chat_history):
    try:
        response = await asyncio.to_thread(pipeline.invoke, {"query": query})
        answer = response["result"]
        sources = "\n".join(
            f"• {doc.metadata['source']} (page {doc.metadata.get('page', 'N/A')})"
            for doc in response["source_documents"]
            if "invoice" in doc.metadata['source'].lower()  # Only include relevant documents
        )

        # Add conversation to chat history
        chat_history.append((query, f"{answer}\n\nSources:\n{sources}"))

        # Save to file
        with open("conversation_history.txt", "a", encoding="utf-8") as file:
            file.write(f"Question: {query}\nAnswer: {answer}\nSources:\n{sources}\n\n")

        return chat_history
    except Exception as e:
        chat_history.append((query, f"Error processing query: {str(e)}"))
        return chat_history



# Function to download the conversation history
def download_conversation():
    with open("conversation_history.txt", "r", encoding="utf-8") as file:
        return file.read()


# Function to clear the chat history
def clear_conversation():
    global conversation_history
    conversation_history = []
    with open("conversation_history.txt", "w", encoding="utf-8") as file:
        file.write("")
    return "Chat history cleared."


# Gradio Interface for Chat UI
def create_interface():
    with gr.Blocks(title="PDF Knowledge Chat Assistant") as interface:
        gr.Markdown("# 📚 PDF Knowledge Chat Assistant\nUpload PDFs and ask questions interactively!")

        with gr.Row():
            with gr.Column():
                file_upload = gr.File(
                    file_count="multiple",
                    label="📂 Upload PDF Documents",
                    type="filepath"
                )
                init_btn = gr.Button("🚀 Initialize System")
                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column():
                chatbot = gr.Chatbot(label="💬 Chat History")
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask something about the uploaded documents...",
                    lines=3
                )
                ask_btn = gr.Button("Ask 🤖")
                download_btn = gr.Button("💾 Download Chat History")
                clear_btn = gr.Button("🗑️ Clear Chat History")

        # State for pipeline and chat history
        pipeline_state = gr.State()
        chat_state = gr.State([])

        # Event handlers
        init_btn.click(
            fn=lambda pdf_files: asyncio.run(async_initialize_rag(pdf_files)),
            inputs=file_upload,
            outputs=pipeline_state
        ).then(
            lambda: "✅ System Ready. You can start asking questions!",
            outputs=status
        ).then(
            lambda: (file_upload.clear(), None),  # Clears previously uploaded files
            inputs=[],
            outputs=[]
        )


        ask_btn.click(
            fn=lambda query, pipeline, chat_history: asyncio.run(
                async_ask_question(query, pipeline, chat_history)
            ),
            inputs=[question_input, pipeline_state, chat_state],
            outputs=chatbot
        )

        download_btn.click(
            fn=download_conversation,
            inputs=[],
            outputs=gr.File(label="📥 Download Conversation History")
        )

        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=status
        )

    return interface


# Launch the Gradio Interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_port=7860,
        share=True,
        show_error=True
    )
