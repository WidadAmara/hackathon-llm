from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation", #task
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

from langchain import PromptTemplate,  LLMChain

template = """
              Write a concise summary of the following text delimited by triple backquotes.
              Return your response in two formats: in 3  bullet points and a single paragraph.

              ```{text}```

              BULLET POINT SUMMARY: 
              

              PARAGRAPH SUMMARY:
              This is where you provide a brief paragraph summarizing the debate.
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

# text = """
# BIDEN: How are you doing, man?

#  TRUMP: How are you doing (ph)?

#  BIDEN: I'm well.

# WALLACE: Gentlemen, a lot of people have been waiting for this night. So let's get going. Our first subject is the Supreme Court.

# President Trump, you nominated Amy Coney Barrett over the weekend to succeed the late Ruth Bader Ginsburg on the court.

# You say the Constitution is clear about your obligation and the Senate's to consider a nominee to the court. Vice President Biden, you say that this is an effort by the president and Republicans to jam through an appointment and what you call an abuse of power.

#  My first question to both of you tonight, why are you right and make the argument you make, and your opponent wrong? And where do you think a Justice Barrett would take the court?

# President Trump, on the first segment you go first. Two minutes.

# TRUMP: Thank you very much, Chris. I will tell you very simply we won the election. Elections have consequences. We have the Senate, we have the White House and we have a phenomenal nominee respected by all; top, top academic. Good in every way. Good in every way.

# In fact, some of her biggest endorsers are very liberal people from Notre Dame and other places. So I think she's going to be fantastic. We have plenty of time. Even if we did it after the election itself, I have a lot of time after the election, as you know.

# So I think that she will be outstanding. She's going to be as good as anybody that has served on that court. We really feel that. We have a professor at Notre Dame, highly respected by all; said she's the single greatest student he's ever had. He's been a professor for a long time at a great school.

# And we just -- we won the election and therefore we have the right to choose her, and very few people knowingly would say otherwise.

# And by the way, the Democrats, they wouldn't even think about not doing it. If they had -- the only difference is they'd try and do it faster. There's no way they would give it up. They had Merrick Garland, but the problem is they didn't have the election. So they were stopped.

# And probably that would happen in reverse also. Definitely it would happen in reverse. So we won the election and we have the right to do it, Chris.

# WALLACE: President Trump, thank you. Same question to you, Vice President Biden. You have two minutes.

# BIDEN: Well, first of all, thank you for doing this and looking forward to this, Mr. President.

# TRUMP: Thank you, Joe.
 
# """

# print(llm_chain.run(text))

 

  # CHAPTER IX.

  # PAGE 140.

  # Questioned J. O’Kane in Dayton; He Beat a Retreat; He Replied in
  # Crawfordsville; Three Resurrections, National, Moral, and Immortal;
  # Conversation in West Lebanon; Everlasting, Forever; Kingdom of
  # God; Sin, Error, Suffering not Endless; In Southern Indiana; Why
  # Live a Christian Life? Bigotry in Breckenridge; Discussion with
  # Mr. Dickerson; Calvinism; Arminianism; Universalism; Debate in
  # Chambersburg.

  # CHAPTER X.

  # PAGE 161.

  # Move to Terre Haute; Lecture in Fort Wayne; A Discussion There; Dr.
  # Thompson; Visit Illinois; Opposition; Discussion in Charleston;
  # Prayed For; Called Infidel; Debate in Green Castle; Conditions of
  # Salvation; God’s Will; All are Spirits; Form of the Teacher Changed;
  # J. Burt and J. H. Jordan, Editors; Oliver Cromwell; Foundation of
  # Character; In Many Places; A Celebration; Meeting in the Rain; Fourth
  # of July Celebration; Debate in Martinsville.

  # CHAPTER XI.

  # PAGE 172.

  # Journey into Northern Illinois; Temperance Lecture; Result of
  # Temperate Drinking; Married; Homeward Bound; High Waters; Difficult
  # Traveling; Trouble in Crossing Streams; A Cold Bath; End of the
  # “Bridal Tour”; A Hard Ride; Debate with E. Kingsbury; In Northern
  # Indiana; Conversation with an Indian; Dark Man and Dark Night;
  # Explanation of Hebrews ix. 27, 28; End of the World; The Earth and
  # Man.

  # CHAPTER XII.

  # PAGE 193.

  # Discussion in Franklin; Justice of God; What the Gospel Is; Society
  # Organized; Discourse on Total Depravity; Conversation with a
  # Presbyterian Minister on Christian Rewards; Talk with a Catholic;
  # A Methodist; A Presbyterian; A Campbellite; Salvation; A Mormon
  # Sermon; Reply to It; A Journey to Louisville and Cincinnati.

  # CHAPTER XIII.

  # PAGE 213.

  # Move to Indianapolis; Extensive Traveling; Henry Ward Beecher; A
  # Fossiled Calvinist; Supposed to be an Orthodox Preacher; Debate
  # in New Philadelphia; Strife Between the North and South; The Old
  # Convention Dead; The New Convention Organized; Discussion in
  # Springfield, Ill.; Abraham Lincoln; God is Love; Is Merciful; Is
  # Just; Is Holy; Travel in Illinois; Conversation with a Presbyterian
  # Clergyman on the Origin of Hell; In Iowa City, and Other Places in
  # Iowa; Home Again; W. J. Chaplin; Discussion with Benjamin Franklin;
  # Debate in Covington; Discussion with Mr. Russell; Publish the “One
  # Hundred and Fifty Reasons”; Review of “Universalism Against Itself”;
  # Publish Another Book; Olive Branch Discontinued; Travel Far and Near.

  # CHAPTER XIV.

  # PAGE 231.

  # Conclude to go to St. Louis; Commence the Golden Era; Association
  # in Crawfordsville; Debate in Dayton; Man in God’s Image; God the
  # Father of All; Man Immortal; Man a Spirit; High Waters; In St. Louis;
  # Why Moved to St. Louis; But Few Friends; First Journey in Missouri;
  # Wet, Hungry, Out in the Cold; In Troy; In Ashley; Four Brothers;
  # In Louisiana; Opposition in London; In Hannibal; Good Friends;
  # Questioned in Palmyra About Slavery; Conversation on Judgment;
  # In Memphis; Questioned; A Presbyterian Preacher Replied; Was to
  # Debate in Newark; Covered with Ice; Missouri River; Discussion in
  # Georgetown; In Southern Missouri; Questioned in Warsaw; In Jefferson
  # City; Hard Work in Danville; Return to St. Louis.

  # CHAPTER XV.

  # PAGE 251.

  # The Golden Era Issued Semi-Monthly; The Missourians; Slave Holders;
  # Travel in Southern Missouri; If Endless Woe is True all Nature would
  # Weep; Region of Iron; Dunkards in Millersville; In Southern Illinois;
  # Philosophy of Christ Being the Savior of the World; Refuse to Debate;
  # Discussion in Carlyle; Inspiration; Our Name; Partialism Approaches
  # Infidelity; Three Downward Steps; Reply to a Sermon; Hayne’s Sermon;
  # Mr. Lewis Debating on his Knees; Written Discussions with two
  # Methodist Ministers; In Northern Missouri; A Preacher Replies; A Log
  # Cabin; Talk with a Slave; Thomas Abbott; Negroes Hung; The Golden
  # Era; Mrs. Manford Lecturing; Let Woman Work; A Circuit in Missouri;
  # Travel in Cold Weather; Debate in Quincy.

  # CHAPTER XVI.

  # PAGE 277.

  # The Golden Era; Extensive Traveling; In Missouri and Kansas; Talk
  # with a Deist in Jefferson City; Moses; The Prophets; Replied to
  # in Pisgah; Talk with a Rum-seller; In Kansas City; In Wyandotte;
  # Conversation with a Clergyman Concerning Christ and his Work;
  # Lectured in Leavenworth; Destruction of Man’s Enemies; In St. Joseph;
  # The Mercy of God; In Kingston; Rich Man and Lazarus.

  # CHAPTER XVII.

  # PAGE 293.

  # The Rebellion Commenced; What Senator Douglas Said; Defenders of our
  # Country; Camp Jackson; Rebel Flag; Great Expectations; Subscribers
  # Lost; Money Lost; All but Two of the Religious Journals Stopped;
  # Could do but Little in Missouri; Society in St. Louis; G. S. Weaver
  # Left; The Unitarian Society; Published Pamphlet on Water Baptism;
  # Discussion with B. H. Smith; Extracts from the Discussion.

  # CHAPTER XVIII.

  # PAGE 315.

  # Discussion in Pontiac; The Apostle’s Faith; His Argument in Romans;
  # Extensive Traveling; In Kansas and Missouri; Price’s Raid; In
  # Ohio and Indiana; Dark Night and Walk in Toledo; Conversation on
  # Destructionism; The Victory; The Death; President Lincoln; Debate in
  # Milford, Ohio; The Restitution an old Doctrine; The Sentiment Wide
  # Spread; At Work in Iowa; Laborers There; Murderers Saved and the
  # Murdered Lost; Intellectual and Moral Growth; What Man Was; What He
  # is to Be; The Victory; Spiritualism; Immoral Preaching; Saved Without
  # Repentance; Preaching a Means of Salvation; A Methodist Minister
  # Believes; The Suicide.

  # CHAPTER XIX.

  # PAGE 346.

  # Last Campaign; In Galesburg, Ill.; The United States Convention;
  # Lombard University; Other Schools; Journey to Missouri; In Macon
  # City; In Brookfield; St. Joseph and Other Cities; Grove Meeting; On
  # the Missouri Bottom; Beautiful Country; Preach in Fillmore and many
  # other Places; Return Home; Anti-Orthodox Preaching; Funeral Sermons;
  # Death; Life; Conclusion.

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
# url = "https://www.gutenberg.org/cache/epub/71224/pg71224.txt"
# response = requests.get(url)
# if response.status_code == 200:
#     data = response.text
url = "summary_project/test-summary.txt"
with open(url, 'r') as file:
     data = file.read()

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=200)
docs = text_splitter.create_documents([data])

print (f"You now have {len(docs)} docs intead of 1 piece of text")

map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

combine_prompt = """

 Write a concise summary in bullet points of the following text delimited by triple backquotes.

 ```{text}```
 BULLET POINT SUMMARY:
 """
combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

summary_chain = load_summarize_chain(llm=llm,
  chain_type='map_reduce',

 map_prompt=combine_prompt_template,
  )

output = summary_chain.run(docs)

print(output)

torch.cuda.empty_cache()


