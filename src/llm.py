# -*- coding: utf-8 -*-
"""
LLM Integration for Hebrew Health Insurance RAG System

Handles prompt building and OpenAI API interaction.
"""

import os
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# System prompt for Hebrew insurance expert
SYSTEM_PROMPT = """You are an expert in private health insurance in Israel. Your role is to answer questions regarding policy terms and insurance coverage based on provided documents.

CRITICAL RULES:
1. Answer ONLY based on the information provided in the Context. The context is in Hebrew, but you must provide your answer in English.
2. Carefully analyze the Hebrew context to find relevant information to answer the user's question.
3. If the information is not present in the context, explicitly state: "I could not find specific information regarding this in the provided documents."
4. Use clear, professional, and friendly English.
5. Be precise regarding amounts, percentages, and conditions - DO NOT hallucinate or invent numbers that do not appear in the context.
6. Clearly state any conditions, limitations, or exclusions mentioned in the coverage.
7. Provide a practical and helpful response to the user.

ANSWER FORMAT:
- A clear and direct answer to the question.
- A detailed breakdown of relevant conditions and financial amounts (converted correctly from the Hebrew text)."""


class InsuranceLLM:
    """
    LLM wrapper for insurance document Q&A.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """
        Initialize the LLM.

        Args:
            model_name: OpenAI model to use
            temperature: LLM temperature (0 = deterministic)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in .env file or environment variables."
            )
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string for LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string with sources
        """
        if not documents:
            return ""

        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = os.path.basename(doc.metadata.get("source", "Unknown Source"))
            page = doc.metadata.get("page", "Unknown Page")
            content_type = doc.metadata.get("content_type", "Unknown Content Type")
            
            context_parts.append(
                f"[מקור {i}: {source}, עמוד {page}, סוג: {content_type}]\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)

    def build_prompt(self, context: str, question: str) -> str:
        """
        Build the user prompt with context and question.

        Args:
            context: Formatted context from retrieved documents
            question: User's question

        Returns:
            Complete user prompt
        """
        if not context:
            return f"""לא נמצא הקשר רלוונטי.

שאלת המשתמש:
{question}

תשובה:"""

        return f"""הקשר (Context):
{context}

שאלת המשתמש:
{question}

תשובה:"""

    def ask(self, question: str, context: str) -> str:
        """
        Ask a question with the provided context.

        Args:
            question: User's question in Hebrew
            context: Formatted context from retrieved documents

        Returns:
            LLM's answer
        """
        # Build user prompt
        user_prompt = self.build_prompt(context, question)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Call LLM
        print("[DEBUG] Sending request to LLM...")
        response = self.llm.invoke(messages)
        print("[DEBUG] Received response from LLM\n")
        
        return response.content
