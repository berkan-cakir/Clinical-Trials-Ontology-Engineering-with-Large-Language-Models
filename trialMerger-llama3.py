from openai import OpenAI
import csv
import os
import time
import datetime
from rdflib import Graph


entitiesOfInterest = ["Biomarker", "Endpoint", "EndpointScore", "Measurement", "MeasurementTool", "Questionnaire", "Timepoint", "OutcomeMeasure", "subClassOf"]
prefixList = set()
synonymList = set()

client = OpenAI(
    base_url = 'http://0.0.0.0:0000/v1',
    api_key='ollama', # required, but unused
)

def timeLog(title):
    try:
        os.mkdir("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/mainOntology/timelogs")
    except Exception:
        pass

    now = datetime.datetime.now()
    print(str(now) + " - " + str(title))

    file = open("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/mainOntology/timelogs/timelog-merge.txt", "a")
    file.write(str(now) + " - " + str(title) + "\n")

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

#get pre-made ontologies and loop over each ontology
def main(mainOntology, path):
    for enum, fileName in enumerate(os.listdir(path)):
        ontologyFile = open(path + fileName, "r").read()
        print(str(enum) + " - working on " + str(fileName))

        #check if file even contains anything
        if validOntology(str(path) + str(fileName)):
            print(enum)
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
                        response = client.chat.completions.create(model="llama3:70b",
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

                        # Add entityOfInterest and its synonyms to synonymList
                        synonymList.add(entityOfInterest)
                        for synonym in synonyms:
                            if synonym not in synonymList:
                                synonymList.add(synonym)

totalPromptTokens = 0
totalCompletionTokens = 0
totalTokens = 0
totalTokenCost = 0
prefixList = set()
synonymList = set()
timeLog("chainedLlama3-70b start")
mainOntology = createMainOntology("chainedLlama3OntologyV1")
main(mainOntology, "/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/chainedOntologiesGPT4v4/")
timeLog("chainedLlama3-70b end")