import json

import docx2txt
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class LabelValuePair(BaseModel):
    id: int
    label: str
    value: str


class Schema(BaseModel):
    label_value_pairs: list[LabelValuePair]


def _read_docx(path) -> str:
    return docx2txt.process(path)


def _get_chain() -> Runnable:
    """
    Requires OPENAI_API_KEY set into the environment
    :return:
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    return llm.with_structured_output(Schema)


def main():
    filepath = "AL-DTC-I-Indication EBC.docx"
    document = _read_docx(filepath)
    llm_chain = _get_chain()

    result: Schema = llm_chain.invoke(document)

    with open('result.json', mode='w', encoding='utf-8') as f:
        f.write(json.dumps(result.model_dump()['label_value_pairs']))


if __name__ == '__main__':
    main()
