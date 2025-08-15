import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import embedding_function
from langchain_community.vectorstores import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ 清空数据库")
        clear_database()

    # 创建或更新数据存储
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # 加载现有数据库
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=embedding_function()
    )

    # 计算分页ID
    chunks_with_ids = calculate_chunk_ids(chunks)

    # 新增或更新文档
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"数据库中已存在的文档: {len(existing_ids)}")

    # 只添加数据库中不存在的文档
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 添加新文档: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ 没有新文档需要添加")


def calculate_chunk_ids(chunks):

    # 这将创建类似“data/monopoly.pdf:6:2”的标识符，即页面来源:页码:块索引

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # 如果页面ID与上一个相同则增加索引
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # 计算块ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # 将其添加到页面元数据中
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def query_db(query_text: str):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function())

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)



if __name__ == "__main__":
    # main()
    query_db("How much total money does a player start with in Monopoly? ")
