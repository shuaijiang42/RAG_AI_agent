from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

#Pandas DataFrame — basically a table in memory that holds your CSV data 
#so you can manipulate it easily.
df = pd.read_csv("ShuAI_permissions_documentation.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        page_content = f"""
        Permission: {row['Permission Name']}
        Description: {row['Description']}
        Notes: {row['Notes']}
        """.strip()

        metadata = {
            "permission_name": row["Permission Name"],
            "module": row["Module/Feature"],
            "default_roles": row["Default Roles"]
        }
        
        document =Document(page_content=page_content, metadata=metadata, id=str(i))
        ids.append(str(i)) #for later manipulating documents and avoiding duplication on re-import
        documents.append(document)

#initialize vector store
vector_store =Chroma(
    collection_name="ShuAI_permissions", # name for this dataset
    persist_directory=db_location,       # where to save the database
    embedding_function=embeddings        # the embedding model I'm using
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)