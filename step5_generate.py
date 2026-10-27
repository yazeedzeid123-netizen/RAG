"""STEP 5 - GENERATE: answer questions with a local Hugging Face model."""
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import LLM_REPO
from step4_retrieve import get_retriever

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful tutor for Class 12 students. Answer ONLY from the context below. "
     "If the answer is not in the context, say 'I could not find this in the notes.'\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'Unknown source')} "
        f"p{d.metadata.get('page', 'unknown')}] {d.page_content}"
        for d in docs
    )


def get_llm():
    model = HuggingFacePipeline.from_model_id(
        model_id=LLM_REPO,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.2,
            "do_sample": True,
            "return_full_text": False,
        },
    )
    return ChatHuggingFace(llm=model)


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
