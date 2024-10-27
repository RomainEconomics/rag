from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.chains.builder import AbstractChainBuilder


##################################################
# QA Chain With OpenAI using context
##################################################


class QAFromContextChain(AbstractChainBuilder):
    model: BaseLanguageModel
    prompt: str | None = None

    def build(self):
        if not self.prompt:
            self.prompt = self._default_template()
        prompt = ChatPromptTemplate.from_template(self.prompt)

        return RunnablePassthrough.assign(prompt=prompt) | RunnablePassthrough.assign(
            response=lambda _: prompt | self.model | StrOutputParser()
        )

    def _default_template(self):
        return """You're an ESG analyst. You're ask to extract information from the provided context.
                Only use the available information to answer the question. Do not add any additional information.

                Question: {question}

                -----------
                Context: {context}
                -----------

                Answer:
                """
