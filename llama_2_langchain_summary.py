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
              Return your response in a single paragraph.

              ```{text}```
              PARAGRAPH SUMMARY:
              This is where you provide a brief paragraph summarizing the debate.
           """

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

text = """
CHAPTER I.

  PAGE 9.

  Parentage and Childhood; “Lord” Timothy Dexter; At School; In
  Haverhill; Shoemaking; Early Aspirations; Converted; Must be a
  Minister; On a Plank; Attend School; A Long Walk; Studies with J. C.
  Waldo and Dr. Cobb; First Preaching; With W. S. Balch; First Tour;
  First Debate; Came out Second Best; Talk with an Englishman about
  American Coarseness; Conversation on Slavery; In Maryland; Talk with
  an Episcopal Clergyman concerning Endless Woe not being Taught in the
  Old Testament; Traveling and Preaching on the Eastern Shore; Return
  to Baltimore; A Storm; Where Truth Flourishes and Where it Does Not;
  Another Location; Self and Faith Abused; Preach in Harper’s Ferry,
  Charleston, Winchester, Va.; A Hard Battle; Cross the Alleghany
  Mountains.

  CHAPTER II.

  PAGE 33.

  In Pittsburg; S. A. Davis, Wife and Daughter; The West; Preach
  in Pennsylvania and Ohio; Western Reserve; Talk with a Bigot;
  Conversation on a Steamboat; Forbidden to Preach; Grave Creek; A
  Mound; My Study; What is Salvation? Proceedings in Bainbridge; Mud;
  In Cincinnati; General Harrison; In Rising Sun; Patriot; Preach in
  Louisville, Ky.; E. M. Pingree; On the Mississippi River; Preach in a
  Steamboat; In New Orleans; Battle Ground.

  CHAPTER III.

  PAGE 47.

  A Sea Voyage; A Meeting at Sea; Tornado; Strange Vessel; In Texas;
  Travel to Houston; Hard Fare; The Country; Sleeping on the Ground;
  Very Thirsty; Must have Water; Colorado River; Sound Asleep on its
  Banks; Cross the River on Logs; Corn Cake; A Surprise; In Houston;
  General Houston; The Attorney-General of Texas; San Jacinto Battle
  Ground; A Pandemonium; Buck Wheat Cakes; Embark for New Orleans; A
  Condemned Vessel; On Allowance; In New Orleans; A Contrast; Ague and
  Fever; Up the Mississippi.

  CHAPTER IV.

  PAGE 55.

  Labors of E. B. Mann; N. Wadsworth; Owner of a Horse; Preach in
  Indiana and Kentucky; A Profane Life; General Clarke; Atheism; The
  Eyeless Fish; A Presbyterian Minister’s Wisdom; No Hell, No Heaven;
  Travel in Ohio; Another Preacher Replies; Labors in Dayton; D. R.
  Biddlecom; George Messenger; R. Smith’s Somersault; J. A. Gurley;
  George Rogers; Start for Indiana; Battle in Harrison; Universalism
  an Old Doctrine, and of God; Partialism an Old Doctrine, but of
  Satan; Grove Meeting; Father St. John: Badly Treated; John O’Kane
  on his Creed; In Indianapolis; A. Longley; A Horse; Questioned by a
  Methodist; In Terre Haute; Very Unpopular.

  CHAPTER V.

  PAGE 74.

  Journey in Ohio; Intemperance; General Baldwin; In Columbus; Death
  Penalty; How to Deal with Offenders; Preach in Newark and Zanesville;
  Hell Discussed; Mrs. Frances D. Gage; Invited to Settle in Marietta;
  W. H. Jolly, In Chillicothe; Opposition in Richmond; --. Webber; In
  Kentucky; Dr. Chamberlin; Opposition in Lexington; Is Universalism
  Infidelity? A Slanderous Story by a D.D.; In Paris; Excursion to
  Patriot; A Discussion; Daniel Parker; Cure the Ague; Good Health.

  CHAPTER VI.

  PAGE 87.

  A Journey East; Talk with a Baptist Minister; Preach in Delaware and
  Centerville, Ohio; W. Y. Emmett; Doors Closed; A. Bond; A. B. Grosh;
  In New England; On the Sea; A Storm; Methodist Preacher Frightened;
  Blow the Trumpet; In Philadelphia; In Delaware; In Pittsburg; Return
  to Cincinnati; Go to Chicago; Bad Roads; In Richmond; Talk with
  a Quaker; A Spirit Returns to Earth; A Spirit Out of the Body; A
  Strange Sight; Preach in God’s Temple; Preach in Chicago; Preach in
  Joliet; Aaron Kinney, an Early Preacher; Bill of Fare; Hard Luck in
  Magnolia; Why Preach; In Hennepin; Political Humbugs; Opposition in
  Washington; Justice of God; In Pekin and Tremont; Frozen; A Preacher
  Replies.

  CHAPTER VII.

  PAGE 103.

  Located in Lafayette; The Christian Teacher Commenced; A Circuit;
  Society Organized; Meeting-house Built; All Alone; Conflict in
  Frankfort; Old Testament Doctrine of Punishment; Debate Proposed
  in Frankfort; Discussion in Independence; Character of my Sermons;
  Slander Refuted; Debate in Burlington; Endless Woe; Some Voting; The
  Use of Discussion; A Traveler.

  CHAPTER VIII.

  PAGE 119.

  Debate in Lafayette; Die in Adam; Alive in Christ; This World and
  World to Come; Battle Ground; In Monticello; A Reply; A Preacher
  Whipped; D. Vines; S. Oyler; I. M. Westfall; B. F. Foster in Indiana;
  Revival Poetry; Ladoga Camp-Meeting; Worship; In Michigan City; An
  Episcopal Preacher; A Wet Ride; Debate in Dayton; Discussion in
  Jefferson; Everlasting Punishment; End of the World; Second Coming of
  Christ; Eternal Life; Meaning of Everlasting.

  CHAPTER IX.

  PAGE 140.

  Questioned J. O’Kane in Dayton; He Beat a Retreat; He Replied in
  Crawfordsville; Three Resurrections, National, Moral, and Immortal;
  Conversation in West Lebanon; Everlasting, Forever; Kingdom of
  God; Sin, Error, Suffering not Endless; In Southern Indiana; Why
  Live a Christian Life? Bigotry in Breckenridge; Discussion with
  Mr. Dickerson; Calvinism; Arminianism; Universalism; Debate in
  Chambersburg.

  CHAPTER X.

  PAGE 161.

  Move to Terre Haute; Lecture in Fort Wayne; A Discussion There; Dr.
  Thompson; Visit Illinois; Opposition; Discussion in Charleston;
  Prayed For; Called Infidel; Debate in Green Castle; Conditions of
  Salvation; God’s Will; All are Spirits; Form of the Teacher Changed;
  J. Burt and J. H. Jordan, Editors; Oliver Cromwell; Foundation of
  Character; In Many Places; A Celebration; Meeting in the Rain; Fourth
  of July Celebration; Debate in Martinsville.

  CHAPTER XI.

  PAGE 172.

  Journey into Northern Illinois; Temperance Lecture; Result of
  Temperate Drinking; Married; Homeward Bound; High Waters; Difficult
  Traveling; Trouble in Crossing Streams; A Cold Bath; End of the
  “Bridal Tour”; A Hard Ride; Debate with E. Kingsbury; In Northern
  Indiana; Conversation with an Indian; Dark Man and Dark Night;
  Explanation of Hebrews ix. 27, 28; End of the World; The Earth and
  Man.

  CHAPTER XII.

  PAGE 193.

  Discussion in Franklin; Justice of God; What the Gospel Is; Society
  Organized; Discourse on Total Depravity; Conversation with a
  Presbyterian Minister on Christian Rewards; Talk with a Catholic;
  A Methodist; A Presbyterian; A Campbellite; Salvation; A Mormon
  Sermon; Reply to It; A Journey to Louisville and Cincinnati.

  CHAPTER XIII.

  PAGE 213.

 
"""

print(llm_chain.run(text))


from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
url = "summary_project/test-summary.txt"
with open(url, 'r') as file:
     data = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
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

