"""STEP 5 - GENERATE: give retrieved chunks + question to the LLM (Hugging Face, free token)."""
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import LLM_REPO, HF_TOKEN
from step4_retrieve import get_retriever

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful tutor for Class 12 students. Answer ONLY from the context below. "
     "If the answer is not in the context, say 'I could not find this in the notes.'\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(f"[{d.metadata['source']} p{d.metadata['page']}] {d.page_content}" for d in docs)


def get_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_REPO, task="text-generation",
        max_new_tokens=512, temperature=0.2,
        huggingfacehub_api_token=HF_TOKEN,
    )
    return ChatHuggingFace(llm=endpoint)


def build_chain(retriever):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT | get_llm() | StrOutputParser()
    )


if __name__ == "__main__":
    retriever = get_retriever()
    chain = build_chain(retriever)
    q = "What is produced in the light reaction?"
    print("Q:", q)
    print("A:", chain.invoke(q))
    print("\nSources:", [d.metadata for d in retriever.invoke(q)])
