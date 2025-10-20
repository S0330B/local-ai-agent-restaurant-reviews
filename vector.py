from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("data/realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

documents: list[Document] = []
ids: list[str] = []

if add_documents:
    for i, row in df.iterrows():
        title = str(row.get("Title", ""))
        review = str(row.get("Review", ""))
        rating = str(row.get("Rating", ""))
        date = str(row.get("Date", ""))

        document = Document(
            page_content=f"{title} {review}",
            metadata={"rating": rating, "date": date},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents and documents:
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"Added {len(documents)} documents to vector store.")
else:
    print("Vector store already exists. Skipping document insertion.")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
print("Retriever ready to use.")
