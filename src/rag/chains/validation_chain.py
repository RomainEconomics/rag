from typing import Type

from pydantic import BaseModel, Field

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.chains.builder import AbstractChainBuilder


##################################################
# Validation Chain (hallucination check)
##################################################


class ValidateExtraction(BaseModel):
    """Based on the context, question and answer, assess if the extraction contains hallucinations or not."""

    hallucination: bool = Field(..., description="True if hallucination is detected")


class ValidationChain(AbstractChainBuilder):
    model: BaseLanguageModel
    tool: Type[BaseModel]
    prompt: str | None = None

    def build(self):
        if not self.prompt:
            self.prompt = self._default_template
        prompt = ChatPromptTemplate.from_template(self.prompt)

        return RunnablePassthrough.assign(
            result_json=lambda inputs: inputs["response"].model_dump_json()
        ) | RunnablePassthrough.assign(
            validation=lambda _: prompt | self.model.with_structured_output(self.tool)
        )

    @property
    def _default_template(self):
        return """ In the context of extraction relevant information from financial and environmental
                reports, you're ask to assess the quality of the extraction.
                You'll be given the context, extracted from reports, the question asked, and the answer.
                Your only job is to assess wether or not the final output from the LLM is relevant and
                does't contain any hallucinations.
                YOU ARE NOT ASKED TO PROVIDE THE ANSWER TO THE QUESTION.

                Question: {question}

                -----------
                Context: {context}
                -----------

                -----------
                Extraction Output: {result_json}
                -----------


                Answer:
                """
