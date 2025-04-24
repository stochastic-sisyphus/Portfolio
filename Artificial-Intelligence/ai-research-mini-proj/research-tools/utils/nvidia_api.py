import openai
from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class OpenAI_LLM(LLM):
    model: str = "gpt-3.5-turbo"

    @property
    def _llm_type(self) -> str:
        return "openai_gpt3.5"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 100),
                temperature=0.2,
                top_p=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise Exception(f"Error calling OpenAI API: {str(e)}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

llm = OpenAI_LLM()

def create_chain(template):
    input_variables = ["input"]
    if "{question}" in template:
        input_variables.append("question")
    prompt = PromptTemplate(input_variables=input_variables, template=template)
    return LLMChain(llm=llm, prompt=prompt)

summarize_chain = create_chain("Summarize the following text:\n\n{input}")
question_chain = create_chain("Generate 3 important questions based on this text:\n\n{input}")
answer_chain = create_chain("Context: {input}\n\nQuestion: {question}\n\nAnswer:")
search_chain = create_chain("Provide some general information about: {input}")
extract_chain = create_chain("Extract relevant information about {input} from the following text:\n\n{input}")

def call_chain(chain, **kwargs):
    try:
        return chain.run(**kwargs)
    except Exception as e:
        print(f"Error calling AI function: {str(e)}")
        return "An error occurred while processing your request."

def summarize_text(text):
    return call_chain(summarize_chain, input=text)

def generate_questions(text):
    return call_chain(question_chain, input=text)

def answer_question(question, context):
    return call_chain(answer_chain, input=context, question=question)

def web_search(query):
    return call_chain(search_chain, input=query)

def extract_information(text, topic):
    return call_chain(extract_chain, input=text)