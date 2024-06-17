import openai
import csv
import os
import time
import datetime
from rdflib import Graph

openai.api_key = ""

entitiesOfInterest = ["Biomarker", "Endpoint", "EndpointScore", "Measurement", "MeasurementTool", "Questionnaire", "Timepoint", "OutcomeMeasure", "subClassOf"]
prefixList = set()
synonymList = set()

def timeLog(title):
    try:
        os.mkdir("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/mainOntology/timelogs")
    except Exception:
        pass

    now = datetime.datetime.now()
    print(str(now) + " - " + str(title))

    file = open("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/mainOntology/timelogs/timelog-merge.txt", "a")
    file.write(str(now) + " - " + str(title) + "\n")

def rateLimitDelay(rpm=3500):
    time.sleep(60/rpm)

def validOntology(ontologyFile):
    try:
        g = Graph()
        g.parse(ontologyFile, format="turtle")
        return True
    except Exception as e:
        print(f"Ontology syntax validation failed. Error: {e}")
        return False

#create main ontology.
def createMainOntology(ontologyName):
    try:
        os.mkdir("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/mainOntology")
    except Exception:
        pass

    return open("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/mainOntology/" + ontologyName + ".owl", "w")

def removeCommentsFromOntologyFile(ontologyFile):
    ontologyFileWithoutComments = ""

    temp = ontologyFile.splitlines()
    for line in temp:
        if len(line) > 0 and line[0] != "#":
            ontologyFileWithoutComments += str(line) + "\n"
    
    return ontologyFileWithoutComments

def getPrefixes(prefixesRaw):
    for prefix in prefixesRaw.split("\n"):
        if " : " not in prefix:
            prefixName = prefix[:prefix.find(":")].split(" ")[1]

            if prefixName not in prefixList:
                prefixList.add(prefixName)
                mainOntology.write(prefix + "\n")
                mainOntology.flush()

def tokenCounter(promptTokens, completionTokens):
    global totalTokens
    global totalPromptTokens
    global totalCompletionTokens
    
    totalPromptTokens += promptTokens
    totalCompletionTokens += completionTokens
    totalTokens += totalPromptTokens + completionTokens

def costCounter(promptTokens, completionTokens, model="gpt-4"):
    global totalTokenCost

    if model == "gpt-3.5-turbo-1106":
        promptPrice = 0.001
        CompletionPrice = 0.002
    elif model == "gpt-4":
        promptPrice = 0.03
        CompletionPrice = 0.06

    tokenCost = ((promptTokens/1000) * promptPrice) + ((completionTokens/1000) * CompletionPrice)

    totalTokenCost += tokenCost
    print(totalTokenCost)

#get pre-made ontologies and loop over each ontology
def main(mainOntology, path):
    for enum, fileName in enumerate(os.listdir(path)):
        ontologyFile = open(path + fileName, "r").read()
        print(str(enum) + " - working on " + str(fileName))

        #check if file even contains anything
        if validOntology(str(path) + str(fileName)):
            #split ontology by newline to get triple
            splittedOntologyFile = ontologyFile.split("\n\n", maxsplit=1)
            prefixesRaw = splittedOntologyFile[0]
            triplesRaw = splittedOntologyFile[1]

            # Remove all comments
            triplesRaw = removeCommentsFromOntologyFile(triplesRaw)

            #Add prefixes if not already added before.
            getPrefixes(prefixesRaw)

            for triple in triplesRaw.split(".\n"):
                #search entityOfInterest in triple
                for entity in entitiesOfInterest:
                    #if entity does exist, split entity by ":".
                    #else, skip.
                    if entity in triple:
                        splittedTriple = triple.split(":")
                        entityOfInterest = splittedTriple[1].split(" ")[0].strip()

                        #send the word after double colon to gpt to generate synonyms
                        response = openai.chat.completions.create(model="gpt-3.5-turbo-1106",
                            messages=[
                                {"role": "system", "content": "You are a synonym generator. Generate all synonyms for the following word/text."},
                                {"role": "assistant", "content": "Only generate the synonyms. So without any extra text or markings."},
                                {"role": "user", "content": entityOfInterest}
                            ],
                            seed=1453,
                            temperature=0)
                        
                        if "," in response.choices[0].message.content:
                            synonyms = response.choices[0].message.content.split(", ")
                        else:
                            synonyms = response.choices[0].message.content

                        #If synonym or word is not in synonymlist, add the word and synonyms to synonymlist and add triple to main ontology. Make sure to also import the correct prefix.
                        #If synonym or word is in synonymlist, skip.
                        if entityOfInterest not in synonymList:
                            print("\t" + str(fileName) + " - " + str(entityOfInterest))
                            mainOntology.write(str(triple) + ".\n")
                            mainOntology.flush()
                        
                        promptToken = response.usage.prompt_tokens
                        completionTokens = response.usage.completion_tokens
                        tokenCounter(promptToken, completionTokens)
                        costCounter(promptToken, completionTokens, "gpt-3.5-turbo-1106")

                        # Add entityOfInterest and its synonyms to synonymList
                        synonymList.add(entityOfInterest)
                        for synonym in synonyms:
                            if synonym not in synonymList:
                                synonymList.add(synonym)

            rateLimitDelay()


totalPromptTokens = 0
totalCompletionTokens = 0
totalTokens = 0
totalTokenCost = 0
timeLog("GPT3Ontology start")
mainOntology = createMainOntology("GPT3Ontology")
main(mainOntology, "/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/ontologiesGPT3v4/")
timeLog("GPT3Ontology end")

totalPromptTokens = 0
totalCompletionTokens = 0
totalTokens = 0
totalTokenCost = 0
prefixList = set()
synonymList = set()
timeLog("chainedGPT3OntologyV4 start")
mainOntology = createMainOntology("chainedGPT3OntologyV4")
main(mainOntology, "/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/chainedOntologiesGPT3v4/")
timeLog("chainedGPT3OntologyV4 end")

totalPromptTokens = 0
totalCompletionTokens = 0
totalTokens = 0
totalTokenCost = 0
prefixList = set()
synonymList = set()
timeLog("GPT4OntologyV4 start")
mainOntology = createMainOntology("GPT4OntologyV4")
main(mainOntology, "/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/ontologiesGPT4v4/")
timeLog("GPT4OntologyV4 end")

totalPromptTokens = 0
totalCompletionTokens = 0
totalTokens = 0
totalTokenCost = 0
prefixList = set()
synonymList = set()
timeLog("chainedGPT4OntologyV4 start")
mainOntology = createMainOntology("chainedGPT4OntologyV4")
main(mainOntology, "/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/chainedOntologiesGPT4v4/")
timeLog("chainedGPT4OntologyV4 end")