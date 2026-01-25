"""
LLM Integration for Hebrew Health Insurance RAG System

Handles prompt building and OpenAI API interaction.
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# System prompt for Hebrew insurance expert
SYSTEM_PROMPT = """אתה מומחה לביטוחי בריאות פרטיים בישראל. תפקידך לענות על שאלות לגבי תנאי הפוליסה והכיסויים הביטוחיים.

כללים חשובים:
1. ענה אך ורק על סמך המידע שמופיע בהקשר (Context) שניתן לך
2. אם המידע לא מופיע בהקשר - אמור "לא מצאתי מידע על כך במסמכים"
3. ציין את מספר העמוד והמקור כשאתה מצטט מידע
4. השתמש בעברית תקינה ומקצועית
5. היה מדויק לגבי סכומים, אחוזים ותנאים - אל תמציא מספרים
6. אם יש תנאים או סייגים לכיסוי - ציין אותם

פורמט התשובה:
- תשובה ברורה וישירה לשאלה
- פירוט התנאים הרלוונטיים
- ציון מקורות (עמוד ושם המסמך)
"""


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
        response = self.llm.invoke(messages)
        
        return response.content
