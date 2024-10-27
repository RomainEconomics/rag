from abc import abstractmethod

from pydantic import BaseModel, ConfigDict

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables import RunnablePassthrough


##################################################
# ChainBuilder
##################################################


class AbstractChainBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def build(self):
        raise NotImplementedError


class ChainBuilder(BaseModel):
    chains: list[AbstractChainBuilder]
    chain: RunnableSerializable | None = None

    def build(self) -> RunnableSerializable:
        full_chain: RunnableSerializable = RunnablePassthrough()

        for chain in self.chains:
            full_chain = full_chain | chain.build()

        self.chain = full_chain

        return self.chain
