import os
import streamlit as st
import pickle
import time
import urllib3
from dotenv import load_dotenv

# Suppress warnings
os.environ["USER_AGENT"] = "NewsQA/1.0"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from a .env file (if present)
load_dotenv()

# LangChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# HuggingFace
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch

# ── Constants ──────────────────────────────────────────────────────────────────
FILE_PATH = "faiss_store_Llama.pkl"
MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_TOKEN  = os.getenv("HF_TOKEN", "")          # set in your environment or .env


# ── Load LLM (cached so it only loads once) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading LLaMA 3.1 — this takes a minute…")
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Check if GPU/CUDA is available; use quantization only if present
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        # Fall back to CPU (slower, but works without GPU)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        repetition_penalty=1.15,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(
        pipeline=pipe,
        pipeline_kwargs={"max_new_tokens": 128},
    )


# ── Prompt template ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Use the context below to answer the question in 2-3 sentences.
Be direct and concise. Do not add notes, comparisons, or extra explanations.
If you don't know the answer, say "I don't know".

Context: {summaries}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["summaries", "question"],
)


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("📰 News Research QA")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
    urls.append(url)

process_clicked = st.sidebar.button("⚙️ Process URLs")
main_placeholder = st.empty()


# ── Process URLs ───────────────────────────────────────────────────────────────
if process_clicked:
    valid_urls = [u for u in urls if u.strip()]

    if not valid_urls:
        st.sidebar.error("Please enter at least one URL.")
    else:
        try:
            main_placeholder.info("Loading articles…")
            loader = WebBaseLoader(valid_urls)
            loader.requests_kwargs = {"verify": False}
            data = loader.load()

            if not data:
                main_placeholder.error("No content loaded. Check your URLs.")
            else:
                main_placeholder.info(f"Loaded {len(data)} page(s). Splitting text…")
                splitter = RecursiveCharacterTextSplitter(
                    separators=["\n\n", "\n", ".", ","],
                    chunk_size=1000,
                )
                docs = splitter.split_documents(data)

                if not docs:
                    main_placeholder.error("No chunks after splitting. Try different URLs.")
                else:
                    main_placeholder.info(
                        f"Split into {len(docs)} chunks. Building embeddings "
                        "(first run downloads ~90 MB — please wait)…"
                    )
                    with st.spinner("Creating embeddings…"):
                        embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            model_kwargs={"device": "cpu"},
                        )
                        vectorstore = FAISS.from_documents(docs, embeddings)

                    with open(FILE_PATH, "wb") as f:
                        pickle.dump(vectorstore, f)

                    main_placeholder.success("✅ Articles processed and saved!")

        except Exception as e:
            main_placeholder.error(f"Error: {e}")


# ── Query ──────────────────────────────────────────────────────────────────────
st.divider()
query = st.text_input("Ask a question about the articles:")

if query:
    if not os.path.exists(FILE_PATH):
        st.warning("Please process URLs first.")
    else:
        with st.spinner("Thinking…"):
            with open(FILE_PATH, "rb") as f:
                vectorstore = pickle.load(f)

            llm = load_llm()

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                combine_prompt=prompt,
            )

            result = chain({"question": query}, return_only_outputs=True)

        st.subheader("Answer")
        st.write(result.get("answer", "No answer found."))

        sources = result.get("sources", "").strip()
        if sources:
            st.subheader("Sources")
            for src in sources.split("\n"):
                if src.strip():
                    st.write(f"- {src.strip()}")