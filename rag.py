import json
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RAGManager:
    def __init__(self, location_file="knowledge_base/locations.json"):
        with open(location_file, "r") as f:
            self.locations = json.load(f)

        docs = [
            f"{loc['name']}: {loc['description']} (x={loc['x']}, y={loc['y']}, θ={loc['theta']})"
            for loc in self.locations
        ]

        self.emb = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        texts = splitter.create_documents(docs)

        self.vectorstore = FAISS.from_documents(texts, self.emb)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 1})

    def retrieve_location(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "No location found"
        content = docs[0].page_content
        # Extract the name from "name: description ..."
        name = content.split(":")[0].strip()
        return name

    def get_location_coords(self, name: str):
        for loc in self.locations:
            if loc["name"].lower() == name.lower():
                return loc
        return None