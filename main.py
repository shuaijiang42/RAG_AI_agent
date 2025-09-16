from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
#specify model
model = OllamaLLM(model="llama3.2")
#model = OllamaLLM(model="phi3")

template = """
You are a helpful AI assistant. Use the information below to answer the question
in a natural, concise way. Avoid listing document IDs or technical details.
Here are some relevant information: {information}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


while True:
    print("\n\n---------------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    information = retriever.invoke(question)
    result = chain.invoke({"information": information, 
                            "question": question})
    print(result)

