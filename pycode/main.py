from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", default="Python")
parser.add_argument("--task", default="Return a list of numbers.")
args = parser.parse_args()

load_dotenv()
llm = OpenAI()

code_prompt = PromptTemplate (
    template="Write a very short {language} function that will {task}.",
    input_variables=["language", "task"]
)

code_chain = LLMChain (
    llm=llm,
    prompt=code_prompt
)

result = code_chain ({
    "language": args.language,
    "task": args.task
})
print(result["text"])