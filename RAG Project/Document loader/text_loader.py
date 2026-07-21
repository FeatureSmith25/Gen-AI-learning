from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel, RunnableLambda
load_dotenv()

model=ChatMistralAI()

prompt=PromptTemplate(
    template='Write a summary for the poem -\n {poem}',
    input_variables=['poem']
)

parser=StrOutputParser()

loader=TextLoader(f'Gen-Ai-Project\RAG Project\Document loader\cricket.txt', encoding='utf-8')
docs=loader.load()
# print(type(docs))
# print(len(docs))
# print(type(docs[0].page_content))
# print(docs[0].metadata)

chain=prompt | model | parser

parallel_chain=RunnableParallel({
    'poem':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x:len(x.split()))
})
# print(chain.invoke({'poem':docs[0].page_content}))

final_chain=chain | parallel_chain
result=final_chain.invoke({'poem':docs[0].page_content})
print(result['poem'])
print(result['word_count'])