import asyncio
import nest_asyncio

from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
    QueryBundle
)
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from hybrid_retriever import HybridRetriever
from utils import TextCleaner, MyFileReader, clean_text
import streamlit as st

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="chroma_database")
chroma_collection = db.get_or_create_collection(
    "resume"
)


def create_text_pipeline():
    """
    创建一个用于文本预处理和嵌入的管道。
    """
    return IngestionPipeline(
        transformations=[TextCleaner(),
                         SentenceSplitter(chunk_size=512, chunk_overlap=10),
                         embed_model]
    )


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = OpenAIEmbedding()
Settings.embed_model = embed_model

def create_resume_index(path="data/resumes"):
    if "resume_index" in st.session_state.keys():
        return

    print(chroma_collection.count())

    documents = SimpleDirectoryReader(input_dir=path).load_data()
    pipeline = create_text_pipeline()
    nodes = pipeline.run(
        documents=documents,
        show_progress=True
    )

    st.session_state.resume_nodes = nodes

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    st.session_state.resume_index = index


def create_retrievers(index, nodes):
    """
    创建用于对简历进行排序的混合检索器。
    """

    print(len(nodes))

    # 使用嵌入技术检索出最相似的前20个节点
    vector_retriever = index.as_retriever(similarity_top_k=20)
    # 使用bm25算法检索出最相似的前20个节点
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
    hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)
    return hybrid_retriever


def create_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-2-v2"):
    """
    创建一个SentenceTransformerRerank
    """

    return SentenceTransformerRerank(top_n=20, model=model_name) if model_name else None


def display_results(data):
    final_list = {}
    final_text = {}
    count = 0
    for i in range(len(data)):
        if data[i].metadata["file_name"] not in final_list:
            final_list[data[i].metadata["file_name"] + ' ' + data[i].id_[-5:]] = data[i].score
            final_text[data[i].metadata["file_name"] + ' ' + data[i].id_[-5:]] = data[i].text
            count += 1
        if count == 10:
            break
    for key in final_list:
        st.write(f"{key} (Score: {final_list[key]:.2f})")
        st.code(final_text[key], language="markdown")
        # st.write(final_text[key])


def resume_ranking_page():

    st.title("简历匹配")
    st.subheader("为您的职位描述获取高度相关的简历。")
    st.write("将显示匹配度最高的前10份简历。")

    job_description = st.text_area("输入工作描述:", height=100)
    job_description = clean_text(job_description)

    with st.spinner("创建简历索引..."):
        create_resume_index(path='D:/rag/Resume_Ranking-main/data/resumes')

    resume_hybrid_retriever = create_retrievers(st.session_state.resume_index, st.session_state.resume_nodes)

    reranker = create_reranker()

    if st.button("对简历进行排序"):
        with st.spinner("处理中..."):
            retrieved_nodes = resume_hybrid_retriever.retrieve(job_description)
            if reranker:
                reranked_nodes = reranker.postprocess_nodes(
                    retrieved_nodes, query_bundle=QueryBundle(job_description)
                )
            else:
                reranked_nodes = retrieved_nodes

        if reranker:
            st.subheader("按分数从高到低排序的最匹配简历:")
        else:
            st.subheader("最匹配的前10份简历:")

        display_results(reranked_nodes)


def main():
    st.set_page_config(page_title="简历匹配服务")

    page = st.sidebar.selectbox("选择服务", ["简历匹配"])

    if page == "简历匹配":
        resume_ranking_page()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)
    nest_asyncio.apply()

    main()
