from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate

from rag.chains.builder import AbstractChainBuilder
from rag.utils import get_base64_image_from_pdf


##################################################
# QA Chain With Image
##################################################


class QAImageChain(AbstractChainBuilder):
    model: BaseLanguageModel
    file_path: str
    # prompt: str | None = None

    def build(self):
        # if not self.prompt:
        #     self.prompt = self._default_template()
        # prompt = ChatPromptTemplate.from_template(self.prompt)

        return RunnablePassthrough.assign(
            pages=lambda inputs: [
                i.metadata["page"] for i in inputs["source_documents"]
            ]
        ) | RunnablePassthrough.assign(
            response=lambda inputs: self._prompt(inputs["pages"])
            | self.model
            | StrOutputParser()
        )

    def _prompt(self, pages: list[int]) -> ChatPromptTemplate:
        base64_images = get_base64_image_from_pdf(self.file_path, pages)

        prompt_messages = [
            SystemMessage(
                content=(
                    "You're ask to use the context or images to extract information from documents."
                    "You're answers must be as precise as possible."
                    "If you don't know the answer, just say you don't know and don't make up an answer"
                    "Use the following pieces of context to answer the question at the end."
                ),
            ),
            self._image_prompt(base64_images),
            HumanMessagePromptTemplate.from_template(self._default_template),
        ]

        return ChatPromptTemplate.from_messages(prompt_messages)

    def _image_prompt(self, data):
        contents = []

        for _, base64_image in data:
            contents += [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]

        return HumanMessage(content=contents)

    @property
    def _default_template(self):
        return """You're an ESG analyst. You're ask to extract information from the provided images.
                Only use the available information to answer the question. Do not add any additional information.

                Question: {question}

                Answer:
                """
