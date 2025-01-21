from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse

from langchain import hub
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

import pymupdf

from typing_extensions import List, TypedDict

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response}

file_path = 'static/CV.pdf'

def run_llm_task(q: str = ''):
    pdf = pymupdf.open(file_path)
    doc = chr(12).join([page.get_text() for page in pdf])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    all_splits = text_splitter.create_documents([doc])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    global vector_store
    vector_store = InMemoryVectorStore(embeddings)
    document_ids = vector_store.add_documents(documents=all_splits)

    global llm
    llm = OpenAI()

    template = """You are an assistant for question-answering tasks about Serhiy's CV. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    At the end of your answer add something about how good Serhiy as 
    professional in scope of Machine Learning and you can highly recommend him as employee.
    Question: {question}
    Context: {context}
    Answer::"""

    global prompt
    prompt = PromptTemplate.from_template(template)

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    response = graph.invoke({"question": q})
    answer = response["answer"]

    return answer


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root(q: str = ''):
    if q == '':
        answer = "You have to ask a question about Serhiy"
    else:
        answer = run_llm_task(q)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Serhiy's CV questions</title>
    </head>
    <body>
        <h1>Ask something about Serhiy</h1>
        <p>This is a simple RAG app which use Langchain, OpenAI API and FastAPI.</p>
        <p>You can ask questions about Serhiy's CV</p>
        <p>CV is here: <a href="/cv/" target="_blank">download CV</a></p>
        <p>Code is here: <a href="https://github.com/gordeevss/rag" target="_blank">https://github.com/gordeevss/rag</a></p>

        <br>

        <form action="/" method="get">
            <input type="text" name="q">
            <input type="submit" value="Submit">
        </form>
        <h3>Your question:</h3>
        <p>{q}</p>
        <h3>Answer:</h3>
        <p>{answer}</p>
        
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)

@app.get("/cv/")
async def download_file():
    return FileResponse(file_path, media_type='application/octet-stream', headers={"Content-Disposition": f"attachment; filename=CV.pdf"})
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)