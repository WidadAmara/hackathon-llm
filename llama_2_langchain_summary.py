# -*- coding: utf-8 -*-
"""Llama_2_langchain_summary.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NwjpR6AClocq1cBOjxoZyrpQbMF_lNx0
"""

#!pip install -q transformers einops accelerate langchain bitsandbytes

#!huggingface-cli login

#!pip install sentencepiece

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer
import transformers
import torch


model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",  # task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=3000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': 1})

template = """
               Write a concise summary of the following debate text delimited by triple backquotes.
            Return your response in 3 bullet points which covers the key points of the text.

              ```{text}```
               BULLET POINT SUMMARY:
            """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

text = """
 BIDEN: How are you doing, man?

 TRUMP: How are you doing (ph)?

 BIDEN: I'm well.

WALLACE: Gentlemen, a lot of people have been waiting for this night. So let's get going. Our first subject is the Supreme Court.

President Trump, you nominated Amy Coney Barrett over the weekend to succeed the late Ruth Bader Ginsburg on the court.

You say the Constitution is clear about your obligation and the Senate's to consider a nominee to the court. Vice President Biden, you say that this is an effort by the president and Republicans to jam through an appointment and what you call an abuse of power.

 My first question to both of you tonight, why are you right and make the argument you make, and your opponent wrong? And where do you think a Justice Barrett would take the court?

President Trump, on the first segment you go first. Two minutes.

TRUMP: Thank you very much, Chris. I will tell you very simply we won the election. Elections have consequences. We have the Senate, we have the White House and we have a phenomenal nominee respected by all; top, top academic. Good in every way. Good in every way.

In fact, some of her biggest endorsers are very liberal people from Notre Dame and other places. So I think she's going to be fantastic. We have plenty of time. Even if we did it after the election itself, I have a lot of time after the election, as you know.

So I think that she will be outstanding. She's going to be as good as anybody that has served on that court. We really feel that. We have a professor at Notre Dame, highly respected by all; said she's the single greatest student he's ever had. He's been a professor for a long time at a great school.

And we just -- we won the election and therefore we have the right to choose her, and very few people knowingly would say otherwise.

And by the way, the Democrats, they wouldn't even think about not doing it. If they had -- the only difference is they'd try and do it faster. There's no way they would give it up. They had Merrick Garland, but the problem is they didn't have the election. So they were stopped.

And probably that would happen in reverse also. Definitely it would happen in reverse. So we won the election and we have the right to do it, Chris.

WALLACE: President Trump, thank you. Same question to you, Vice President Biden. You have two minutes.

BIDEN: Well, first of all, thank you for doing this and looking forward to this, Mr. President.

TRUMP: Thank you, Joe.
"""

print(llm_chain.run(text))

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
# # url = "https://www.gutenberg.org/cache/epub/71224/pg71224.txt"
# # response = requests.get(url)
# # if response.status_code == 200:
# #     data = response.text
# url = "/home/widad/summary_project/test-summary.txt"
# with open(url, 'r') as file:
#      data = file.read()

# text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=150)
# docs = text_splitter.create_documents([data])

# print (f"You now have {len(docs)} docs intead of 1 piece of text")

# map_prompt = """
# Write a concise summary of the following:
# "{text}"
# CONCISE SUMMARY:
# """
# map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

# combine_prompt = """

#  Write a concise summary in bullet points of the following text delimited by triple backquotes.

#  ```{text}```
#  BULLET POINT SUMMARY:
#  """
# combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

# summary_chain = load_summarize_chain(llm=llm,
#   chain_type='map_reduce',

#  map_prompt=combine_prompt_template,
#   )

# output = summary_chain.run(docs)

# print(output)

# torch.cuda.empty_cache()

