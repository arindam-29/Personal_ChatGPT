import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.model_loader import ModelLoader
from exception.custom_exception import ProjectCustomException
from logger import GLOBAL_LOGGER as loger   
from utils.qdrant_vector_db import QdrantVDB
from prompt.prompt_metadata import PromptType
from prompt.prompt_library import PROMPT_REGISTRY

class ConversationalRAG:
    """
    LCEL-based Conversational RAG with lazy retriever initialization.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        rag.load_retriever_from_faiss(index_path="faiss_index/abc", k=5, index_name="index")
        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, user_name: str):
        try:
            self.user_name = user_name
            self.session_id = "Arindam_session"  # Placeholder for session management

            # Load LLM and prompts once
            modelload = ModelLoader()
            self.embeddings = modelload.load_embeddings()
            self.llm = modelload.load_llm()

            qdrant_ds = QdrantVDB()
            vector_store = qdrant_ds.get_vector_store(self.embeddings, collection_name=user_name)
            self.retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1})

            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            loger.info("ConversationalRAG initialized", user_name=self.user_name)
        except Exception as e:
            loger.error("Failed to initialize ConversationalRAG", error=str(e))
            raise ProjectCustomException("Initialization error in ConversationalRAG", sys)

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise ProjectCustomException(
                    "RAG chain not initialized. Call load_retriever_from_faiss() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                loger.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."
            loger.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            loger.error("Failed to invoke ConversationalRAG", error=str(e))
            raise ProjectCustomException("Invocation error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise ProjectCustomException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
            )

            # 2) Retrieve docs for rewritten question
            retrieve_docs = question_rewriter | self.retriever | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            loger.info("LCEL graph built successfully", session_id=self.session_id)
        except Exception as e:
            loger.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise ProjectCustomException("Failed to build LCEL chain", sys)

if __name__ == "__main__":
    rag = ConversationalRAG(user_name="Arindam")
    test_chat_history = []
    # test_question = "How many years of experience Arindam has?"
    test_question = "Give me Rudy's vaccine report?"
    answer = rag.invoke(user_input=test_question, chat_history=test_chat_history)
    print("Answer:", answer)