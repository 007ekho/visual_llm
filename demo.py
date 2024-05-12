import os
import openai
import streamlit as st
from pathlib import Path
from PIL import Image
from sql_execution import execute_sf_query
# from sql_execution import execute_df_query
from sql_execution import get_completion
from langchain.prompts import load_prompt
# from langchain import OpenAI, LLMChain
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import matplotlib.pyplot as plt
import seaborn as sn


OPENAI_API_KEY=st.secrets.OPENAI_API_KEY
# root_path = [p for p in Path(__file__).parents if p.parts[-1]=="LLMSQL"][0]
root_path = Path(__file__).resolve().parent

#create front end
st.title("AI sql assistant")
user_input = st.text_input("enter your question here")
tab_title = ["result","Query","plot","test"]
tabs = st.tabs(tab_title)

#upload the image
# erd_image = Image.open(f'{root_path}/eth-output.jpg')
# with tabs[2]:
#     st.image(erd_image)

#create prompt
prompt_template = load_prompt("C:/Users/USER/Downloads/llmsql/prompts/tpch_prompt.yaml")
llm = OpenAI(temperature=0)

sql_generation_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
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
        print(completion)
        st.write(exec(completion))
        



        # llm = OpenAI(temperature=0)
        # template = """give this answer {result_list} write a statment to address the question {user_input}"""
        # prompt = PromptTemplate(input_variables=["result_list","user_input"],template=template)
        # response =LLMChain(llm=llm,prompt=prompt, verbose=True)
        # st.write(response)



