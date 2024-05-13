import os
import openai
import streamlit as st
from pathlib import Path
from PIL import Image
from langchain import hub
from sql_execution import execute_sf_query
# from sql_execution import execute_df_query
from sql_execution import get_completion
from langchain.prompts import load_prompt
# from langchain import OpenAI, LLMChain

from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
# from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import requests
import yaml
# import seaborn as sn


OPENAI_API_KEY=st.secrets.OPENAI_API_KEY
# root_path = [p for p in Path(__file__).parents if p.parts[-1]=="LLMSQL"][0]
# root_path = Path(__file__).resolve().parent

#create front end
st.title("CHAT WITH YOUR SNOWFLAKE DATABASE")
user_input = st.text_input("enter your question here")
tab_title = ["result","Query","plot","test"]
tabs = st.tabs(tab_title)

# def load_prompt_from_github(file_url):
#     response = requests.get(file_url)
#     response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
#     prompt_data = yaml.safe_load(response.text)
#     return prompt_data

# # Example GitHub raw file URL
# github_raw_url = "https://raw.githubusercontent.com/007ekho/visual_llm/main/tpch_prompt.yaml"

# Load prompt from GitHub
# template = load_prompt_from_github(github_raw_url)

#create prompt
# prompt_template = load_prompt("https://github.com/007ekho/visual_llm/blob/main/tpch_prompt.yaml")
prompt = hub.pull("ehi-123/bs")
llm = OpenAI(temperature=0)

sql_generation_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
# python_generation_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

if user_input:
    sql_query = sql_generation_chain(user_input)
    result = execute_sf_query(sql_query['text'])

    with tabs[0]:
        st.write(result)
    with tabs[1]:
        st.write(sql_query['text'])
    with tabs[2]:
        #result_list =result.astype(str).apply(' '.join, axis=1).tolist()
        #prompt = "give this answer {result_list} write a statment to address the question {user_input}"
        sql_query = sql_generation_chain(user_input)
        result = execute_sf_query(sql_query['text'])
        labels = result.iloc[:, 0]
        values = result.iloc[:, 1]
        fig, ax = plt.subplots()
        ax.bar(labels, values)
        st.pyplot(fig)
    with tabs[3]:
        data_dict = result.to_dict()
        prompt = "given this details {data_dict} generate a complete python  code using matplotlib and seaborn to provide the best visualization for  this details"

        # Call the function to get completion
        completion=get_completion(prompt, result, openai)
        
        st.write(completion)
        



        # llm = OpenAI(temperature=0)
        # template = """give this answer {result_list} write a statment to address the question {user_input}"""
        # prompt = PromptTemplate(input_variables=["result_list","user_input"],template=template)
        # response =LLMChain(llm=llm,prompt=prompt, verbose=True)
        # st.write(response)



