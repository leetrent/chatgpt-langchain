from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()
llm = OpenAI()

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="Python")
parser.add_argument("--task", default="Return a list of numbers.")
args = parser.parse_args()

code_prompt = PromptTemplate (
    template="Write a very short {language} function that will {task}.",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate (
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}."

)

code_chain = LLMChain (
    llm=llm,
    prompt=code_prompt,
    output_key="code"
)

test_chain = LLMChain(
    llm=llm,
    prompt=test_prompt,
    output_key="test"
)

combined_chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"]
)

result = combined_chain ({
    "language": args.language,
    "task": args.task
})

print("\n\n>>>>>>Generated Code:")
print(result["code"])
print("\n\n>>>>>Generated Test:")
print(result["test"])