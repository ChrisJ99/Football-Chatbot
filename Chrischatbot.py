#######################################################
#  Initialise NLTK Inference
#######################################################
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
read_expr = Expression.fromstring 

#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas as pd #Pandas is used for creating dataframes and gives tools for reading and writing data within python
import requests #Requests is used sending https requests, in this case its used for the football API
import json 
import wikipedia #Wikipedia API 
import csv
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 


lemmer = nltk.stem.WordNetLemmatizer()


#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
data1=pd.read_csv('FootballQA.csv')
question=data1['Q'].tolist()
answer1=data1['A'].tolist()
kb=[]

data = pd.read_csv('kb.csv', header=None) #This writes the data in the knowledgebase into a dataframe using pandas
[kb.append(read_expr(row)) for row in data[0]] #Sets the kb append command up
r=ResolutionProver().prove(None, kb) #Checks if there is a contridiction in the logic inside the knowledge base before running the bot.
if (r==True): #THE BOT WILL NOT RUN IF THERE IS A CONTRADICTION
    sys.exit("There is a contradiction in the knowledge base please correct it and try again!") 
    
    

##Setting up the data table from the football API

url = "https://footballapi.pulselive.com/football/players?pageSize=1000&compSeasons=418&altIds=true&page=0&type=player&id=-1&compSeasonId=418"
payload={} #Football API - Page size needs to be set as the way the premier league site handles the request is odd
headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:87.0) Gecko/20100101 Firefox/87.0', #These are the headers that were produced by postman
  'Accept': '*/*',
  'Accept-Language': 'en-GB,en;q=0.5',
  'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
  'Origin': 'https://www.premierleague.com',
  'Connection': 'keep-alive',
  'Referer': 'https://www.premierleague.com/',
  'If-None-Match': 'W/0e33790ed3ba724bd4740e1d56fa739d9',
  'TE': 'Trailers'
}
r = requests.request("GET", url, headers=headers, data=payload) #uses requests to import the json data from the api
playerdata = r.json()
df = pd.json_normalize(playerdata['content']) #Uses the data from the API to fill a data table


#######################################################
#  Initialise AIML agent
#######################################################
import aiml
# Create a Kernel object. No string encoding (all I/O is unicode)
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="chris-logic.xml")


#######################################################
# Welcome user
#######################################################
print("Welcome to the chat bot. Please feel free to ask me football related questions!")

#######################################################
# Main loop
#######################################################
while True:
    #get user input
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break #break stops the loop here and therefore ends the chat bot...
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml': #aiml defines the responses that can be entered by the user
        answer = kern.respond(userInput) #answer dictates the users input at each stage
    #post-process the answer for commands
    
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(params[1])
            break
        elif cmd == 1: #Wikipedia API
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=False)
                print(wSummary)
            except:
                print("Sorry, I do not know that. Be more specific!")
        elif cmd == 2: #This command will display the players that play for the entered team
            team = df.loc[df['currentTeam.club.name'] == (params[1])] #Searches the datatable for the  and selects the rows that match!
            playerNames = team['name.display']
            if answer:
                if team.empty == True:
                    print("Incorrect club name please try again (Club names are case sensitive!)")
                else:
                    print(playerNames.to_string(index=False))
        elif cmd == 3: #WHO DOES * PLAY FOR - will display the team a player plays for based off user input
            playername = df.loc[df['name.display'] == (params[1])]
            teamString = playername['currentTeam.club.name']
            if answer:
                if playername.empty == True:
                    print("Incorrect player name please try again (names are case sensitive!)")
                else:
                    print(teamString.to_string(index=False))
        elif cmd == 4: #HOW OLD IS * - will display the age of the entered player
            agecheck = df.loc[df['name.display'] == (params[1])]
            ageString = agecheck['age']

            if answer:
                if agecheck.empty == True:
                    print("Incorrect player name please try again (names are case sensitive!)")
                else:
                    print(ageString.to_string(index=False))
        elif cmd == 5: #WHAT POSITION DOES * PLAY - Displays the position the entered player plays
            position = df.loc[df['name.display'] == (params[1])]
            positionCheck = position['info.positionInfo']
            if answer:
                if position.empty == True:
                    print("Incorrect player name please try again (names are case sensitive!)")
                else:
                    print(positionCheck.to_string(index=False))
                    

        elif cmd == 50: #When the user inputs something that is not writted in AIMl form this is the command that will run
            def responses(user): #This uses cosine similarity and compares the input to the csv containing some question and answer pairs.
                response=''
                question.append(user) 
                TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english') #Converts a collection of raw documents to a matrix of TF-IDF features.
                tfidf = TfidfVec.fit_transform(question) #Stop words are so commonly used that they carry very little useful information.
                val = cosine_similarity(tfidf[-1], tfidf) 
               
                id1=val.argsort()[0][-2]
                flat = val.flatten()
                flat.sort()
                req = flat[-2]
                
                try:
                    if(req==0):
                        robo_response=response+"I am sorry! I don't understand you"
                        return robo_response
                    else:
                        response = response+answer1[id1]
                        question.remove(user)
                        return response      
                except:        
                    print("I am sorry! I don't understand you")
            if input:
                print(responses(str(params[1])))
            else:
                print("Error")
            

        # Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is from *" or "I know that * plays for *"
            object,subject=params[1].split(' is ') #splits the input, subject and object are the spliced inputs.
            exprTrue=read_expr((subject + '(' + object + ')').lower()) #this expression when returns true means that the input does not exist in or contradict the knowledge base
            exprFalse=read_expr(('-'+subject + '(' + object + ')').lower()) #when the expression query returns false this means the expression must already exitst in KB()
            answerTrue=ResolutionProver().prove(exprTrue, kb) #Resolution prover(Query, dataset), this checks the query through the knowledge base
            answerFalse=ResolutionProver().prove(exprFalse, kb)
            if answerFalse:
                print("There is goes against something i already know!")
            else:
                kb.append(exprTrue) 
                print('OK, I will remember that',object,'is', subject) 
                              
        elif cmd == 32: # if the input pattern is "check that * is from *"
            object,subject=params[1].split(' is ')
            expr=read_expr((subject + '(' + object + ')').lower())
            expr1=read_expr(('-'+subject + '(' + object + ')').lower())
            answer=ResolutionProver().prove(expr, kb)
            answer1=ResolutionProver().prove(expr1, kb)
            if answer:
               print('Correct.')
            elif answer1:
               print('It is not true.') 
            else: 
               print('I cannot say for sure whether that is true')
        
        elif cmd == 33: # CHECK THE DETAILS IF * PLAYS FOR *
            object,subject=params[1].split(' is ')
            expr=read_expr(('plays_for' + '(' + object + ',' + subject + ')').lower())
            answer=ResolutionProver().prove(expr, kb)
            if answer:
               print('Correct.')
            else: 
               print('I cannot say for sure whether that is true')



    else:
        print(answer)
