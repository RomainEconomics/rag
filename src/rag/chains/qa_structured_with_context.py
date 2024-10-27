from typing import Type

from pydantic import BaseModel

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from rag.chains.builder import AbstractChainBuilder


##################################################
# QA Chain With OpenAI using structured output
##################################################


class StructuredOutputFromContextChain(AbstractChainBuilder):
    model: BaseLanguageModel
    tool: Type[BaseModel]
    prompt: str | None = None

    def build(self):
        if not self.prompt:
            self.prompt = self._default_template()
        prompt = ChatPromptTemplate.from_template(self.prompt)

        return RunnablePassthrough.assign(prompt=prompt) | RunnablePassthrough.assign(
            response=lambda _: prompt | self.model.with_structured_output(self.tool)
        )

    def _default_template(self):
        return """You're an ESG analyst. You're ask to extract information from the provided context.
                Only use the available information to answer the question. Do not add any additional information.
                Use the appropriate tool to extract the information.

                Question: {question}

                -----------
                Context: {context}
                -----------

                Answer:
                """
