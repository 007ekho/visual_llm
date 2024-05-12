import snowflake.connector
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from langchain.prompts import load_prompt
# from langchain import OpenAI, LLMChain
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
# from app_secrets import *

def execute_sf_query(sql):
    # Snowflake connection parameters
    

    query=sql

    try:
        # Establish a connection to Snowflake
        # conn = snowflake.connector.connect("myconnection_test")
        conn = st.connection("snowflake")
        # Create a cursor object
        cur = conn.cursor()

        # Execute the query
        try:
            cur.execute(query)
            print(cur.execute(query))
        except snowflake.connector.errors.ProgrammingError as pe:
            print("Query Compilation Error:", pe)
            return("Query compilation error")

        # Fetch all results
        query_results = cur.fetchall()

        # Get column names from the cursor description
        column_names = [col[0] for col in cur.description]

        # Create a Pandas DataFrame
        data_frame = pd.DataFrame(query_results, columns=column_names)
        
        return data_frame

    except snowflake.connector.errors.DatabaseError as de:
        print("Snowflake Database Error:", de)

    except Exception as e:
        print("An error occurred:", e)

    finally:
        # Close the cursor and connection
        try:
            cur.close()
        except:
            pass

        try:
            conn.close()
        except:
            pass
# def execute_py():


# def execute_df_query(data_frame):
#     prompt_template2 = load_prompt("C:/Users/USER/Downloads/llmsql/python_pt.yaml")
#     llm = OpenAI(temperature=0)
#     data_dict = data_frame.to_dict()
    
#     # Generate statement
#     statement = f"given this details {data_dict} suggest a matplotlib code to visualize this details"
#     print(statement)
#     python_generation_chain = LLMChain(llm=llm, prompt=prompt_template2, verbose=True)
#     python_query = python_generation_chain(statement)
#     print(python_query)
#     result = python_query['text']
#     st.write(result)


def get_completion(prompt, data_frame, openai, model="gpt-3.5-turbo"):
    # Replace placeholders in the prompt with actual values
    prompt = prompt.format(data_dict=data_frame.to_dict())
    messages= [{"role": "user", "content": prompt}]
    # Create a completion
    response =openai.chat.completions.create(
        model=model,
        messages= messages,
        max_tokens=400,
        temperature=0
    )
    res =response.choices[0].message.content

    return res

if __name__ == "__main__":
    # Snowflake query
    query = '''
            SELECT COUNT(DISTINCT bus_number) AS num_buses FROM bus_journey;
    '''
    execute_sf_query(query)