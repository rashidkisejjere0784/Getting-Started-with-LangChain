import os
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "API KEY"

flan_t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xl",
    model_kwargs = {"temperature": 1e-5}
)

template = """Question: {question}

Answer: """

prompt = PromptTemplate(template = template, input_variables = ["question"])

llm_chain = LLMChain(
    prompt = prompt,
    llm = flan_t5
)

question = "Who is the first person to land on the moon"

print(llm_chain.run(question))