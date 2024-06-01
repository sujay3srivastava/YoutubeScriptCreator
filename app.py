import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
# Load environment variables from the .env file
load_dotenv()
import sys
print("Python version")
print(sys.version)
# Access the API key
#API_KEY = os.getenv('API_KEY')
os.environ['OPENAI_API_KEY'] = st.secrets['API_KEY']



#App framework
st.title('🦜️🔗Youtube Title Creator')
prompt = st.text_input('Plug in your prompt here')

#PromptTemplate
titletemplate = PromptTemplate(
    input_variables = ['topic'],
    template= 'write me a youtube video title about {topic}'
)

scripttemplate = PromptTemplate(
    input_variables = ['title','wikipedia_research'],
    template= 'write me a youtube video script based on this title TITLE :{title} while using this wikipedia research: {wikipedia_research}')

#memory
titlememory = ConversationBufferMemory(input_key = 'topic', memory_key =' chat_history' )
scriptmemory = ConversationBufferMemory(input_key = 'title', memory_key =' chat_history' )
#LLMs

llm = OpenAI(model_name="gpt-3.5-turbo-0125",temperature=0.9)
titlechain = LLMChain(llm= llm, prompt = titletemplate, verbose = True, output_key = "title",memory = titlememory)
scriptchain = LLMChain(llm= llm, prompt = scripttemplate, verbose = True,  output_key = "script", memory = scriptmemory)
#sequentialchain = SequentialChain(chains = [titlechain,scriptchain], input_variables = ["topic"],
#                                  output_variables = [ "title","script"], verbose = True)
wiki = WikipediaAPIWrapper()
#show stuff on the screen

if prompt:
    title = titlechain.run(prompt)
    wikiresearch = wiki.run(prompt)
    script = scriptchain.run(title = title, wikipedia_research = wikiresearch)
    #response = llm(prompt)
   # response = sequentialchain({"topic": prompt}, return_only_outputs=True)
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(titlememory.buffer)
    with st.expander('Script History'):
        st.info(scriptmemory.buffer)
    with st.expander('Wikipedia Research History'):
        st.info(wikiresearch)
