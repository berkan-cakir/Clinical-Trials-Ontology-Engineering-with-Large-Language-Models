import openai
import csv
import os
import time
import datetime
openai.api_key = ""

def getClinicalTrails(filePath='ctg-studiesV3.csv', delimiter=','):
    clinicalTrails = []

    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        for enum, row in enumerate(csv_reader):
            clinicalTrails.append(row)
    
    return clinicalTrails

def rateLimitDelay(rpm=3500):
    time.sleep(60/rpm)

def timeLog(title):
    try:
        os.mkdir("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/timelogs")
    except Exception:
        pass

    now = datetime.datetime.now()
    print(str(now) + " - " + str(title))

    file = open("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/timelogs/timelog.txt", "a")
    file.write(str(now) + " - " + str(title) + "\n")

def promptGPT3(clinicalTrail):
    NCT = str(clinicalTrail[0])
    mainOutcomes = clinicalTrail[1]
    secondaryOutcomes = clinicalTrail[2]

    #gpt-3.5-turbo-1106
    response = openai.chat.completions.create(model="gpt-3.5-turbo-1106",
                                        messages=[
                                            {"role": "system", "content": "You are a computer scientist tasked with creating an ontology from clinical trails in the OWL ontology code format."},
                                            {"role": "assistant", "content": "You have to first extract the biomarkers, endpoint scores, outcome measurement tools, and questionaire types from cinical trail and then turn that into the owl ontology code format" + NCT},
                                            {"role": "assistant", "content": "The only thing you have to do is put the biomarkers, endpoint scores, outcome measurement tools, and questionaires as subclasses of their respective mainclasses (i.e. ex:Biomarker, ex:EndpointScore, ex:MeasurementTool, and ex:Questionnaire)."},
                                            {"role": "assistant", "content": """Example: @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
                                                                                        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                                                                                        @prefix owl: <http://www.w3.org/2002/07/owl#> .
                                                                                        @prefix ex: <http://example.org/frailty#> .

                                                                                        # Main classes
                                                                                        ex:MeasurementTool a owl:Class .
                                                                                        ex:Questionnaire a owl:Class .
                                                                                        ex:Biomarker a owl:Class .
                                                                                        ex:EndpointScore a owl:Class .

                                                                                        # Subclasses for biomarkers
                                                                                        ex:Exhaustion rdfs:subClassOf ex:Biomarker .
                                                                                        ex:UnintentionalWeightLoss rdfs:subClassOf ex:Biomarker .
                                                                                        ex:PhysicalActivity rdfs:subClassOf ex:Biomarker .
                                                                                        ex:GripStrength rdfs:subClassOf ex:Biomarker .
                                                                                        ex:WalkingSpeed rdfs:subClassOf ex:Biomarker .

                                                                                        # Subclasses for measurement tools
                                                                                        ex:CESD rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:MinnesotaLeisureTimeActivityQuestionnaire rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:Dynamometer rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:BalanceTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:GaitSpeedTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:ChairStandsTest rdfs:subClassOf ex:MeasurementTool .

                                                                                        # Subclasses for endpoint scores
                                                                                        ex:HbA1cChange rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBBalanceScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBGaitSpeedScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBChairStandsScore rdfs:subClassOf ex:EndpointScore ."""},
                                            {"role": "user", "content": mainOutcomes + " " + secondaryOutcomes}
                                        ],
                                        seed=1453,
                                        temperature=0)
    
    return NCT, response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

def chainedPromptGPT3(clinicalTrail):
    NCT = str(clinicalTrail[0])
    mainOutcomes = clinicalTrail[1]
    secondaryOutcomes = clinicalTrail[2]
    promptTokens = 0
    completionTokens = 0

    response1 = openai.chat.completions.create(model="gpt-3.5-turbo-1106",
                                        messages=[
                                            {"role": "system", "content": "You are a biologist tasked with extracting biomarkers, endpoint scores, outcome measurement tools, and questionaire types."},
                                            {"role": "user", "content": mainOutcomes + " " + secondaryOutcomes}
                                        ],
                                        seed=1453,
                                        temperature=0)
    promptTokens += response1.usage.prompt_tokens
    completionTokens += response1.usage.completion_tokens

    response2 = openai.chat.completions.create(model="gpt-4",
                                        messages=[
                                            {"role": "system", "content": "You are a computer sciencetist tasked with creating a ontology from clinical trails in the OWL ontology code format."},
                                            {"role": "assistant", "content": "You have to convert the biomarkers, endpoint scores, outcome measurement tools, and questionaire types into an ontology."},
                                            {"role": "assistant", "content": "The only thing you have to do is put the biomarkers, endpoint scores, outcome measurement tools, and questionaires as subclasses of their respective mainclasses (i.e. ex:Biomarker, ex:EndpointScore, ex:MeasurementTool, and ex:Questionnaire)."},
                                            {"role": "assistant", "content": """Example: @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
                                                                                        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                                                                                        @prefix owl: <http://www.w3.org/2002/07/owl#> .
                                                                                        @prefix ex: <http://example.org/frailty#> .

                                                                                        # Main classes
                                                                                        ex:MeasurementTool a owl:Class .
                                                                                        ex:Questionnaire a owl:Class .
                                                                                        ex:Biomarker a owl:Class .
                                                                                        ex:EndpointScore a owl:Class .

                                                                                        # Subclasses for biomarkers
                                                                                        ex:Exhaustion rdfs:subClassOf ex:Biomarker .
                                                                                        ex:UnintentionalWeightLoss rdfs:subClassOf ex:Biomarker .
                                                                                        ex:PhysicalActivity rdfs:subClassOf ex:Biomarker .
                                                                                        ex:GripStrength rdfs:subClassOf ex:Biomarker .
                                                                                        ex:WalkingSpeed rdfs:subClassOf ex:Biomarker .

                                                                                        # Subclasses for measurement tools
                                                                                        ex:CESD rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:MinnesotaLeisureTimeActivityQuestionnaire rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:Dynamometer rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:BalanceTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:GaitSpeedTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:ChairStandsTest rdfs:subClassOf ex:MeasurementTool .

                                                                                        # Subclasses for endpoint scores
                                                                                        ex:HbA1cChange rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBBalanceScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBGaitSpeedScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBChairStandsScore rdfs:subClassOf ex:EndpointScore ."""},
                                            {"role": "user", "content": response1.choices[0].message.content},
                                        ],
                                        seed=1453,
                                        temperature=0)
    promptTokens += response2.usage.prompt_tokens
    completionTokens += response2.usage.completion_tokens

    return NCT, response2.choices[0].message.content, promptTokens, completionTokens

def promptGPT4(clinicalTrail):
    NCT = str(clinicalTrail[0])
    mainOutcomes = clinicalTrail[1]
    secondaryOutcomes = clinicalTrail[2]

    response = openai.chat.completions.create(model="gpt-4",
                                        messages=[
                                            {"role": "system", "content": "You are a computer scientist tasked with creating an ontology from clinical trails in the OWL ontology code format."},
                                            {"role": "assistant", "content": "You have to first extract the biomarkers, endpoint scores, outcome measurement tools, and questionaire types from cinical trail and then turn that into the owl ontology code format" + NCT},
                                            {"role": "assistant", "content": "The only thing you have to do is put the biomarkers, endpoint scores, outcome measurement tools, and questionaires as subclasses of their respective mainclasses (i.e. ex:Biomarker, ex:EndpointScore, ex:MeasurementTool, and ex:Questionnaire)."},
                                            {"role": "assistant", "content": "Make sure to include the prefixes as shown in the example"},
                                            {"role": "assistant", "content": """Example: @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
                                                                                        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                                                                                        @prefix owl: <http://www.w3.org/2002/07/owl#> .
                                                                                        @prefix ex: <http://example.org/frailty#> .

                                                                                        # Main classes
                                                                                        ex:MeasurementTool a owl:Class .
                                                                                        ex:Questionnaire a owl:Class .
                                                                                        ex:Biomarker a owl:Class .
                                                                                        ex:EndpointScore a owl:Class .

                                                                                        # Subclasses for biomarkers
                                                                                        ex:Exhaustion rdfs:subClassOf ex:Biomarker .
                                                                                        ex:UnintentionalWeightLoss rdfs:subClassOf ex:Biomarker .
                                                                                        ex:PhysicalActivity rdfs:subClassOf ex:Biomarker .
                                                                                        ex:GripStrength rdfs:subClassOf ex:Biomarker .
                                                                                        ex:WalkingSpeed rdfs:subClassOf ex:Biomarker .

                                                                                        # Subclasses for measurement tools
                                                                                        ex:CESD rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:MinnesotaLeisureTimeActivityQuestionnaire rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:Dynamometer rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:BalanceTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:GaitSpeedTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:ChairStandsTest rdfs:subClassOf ex:MeasurementTool .

                                                                                        # Subclasses for endpoint scores
                                                                                        ex:HbA1cChange rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBBalanceScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBGaitSpeedScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBChairStandsScore rdfs:subClassOf ex:EndpointScore ."""},
                                            {"role": "user", "content": mainOutcomes + " " + secondaryOutcomes}
                                        ],
                                        seed=1453,
                                        temperature=0)
    
    return NCT, response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

def chainedPromptGPT4(clinicalTrail):
    NCT = str(clinicalTrail[0])
    mainOutcomes = clinicalTrail[1]
    secondaryOutcomes = clinicalTrail[2]
    promptTokens = 0
    completionTokens = 0

    response1 = openai.chat.completions.create(model="gpt-4",
                                        messages=[
                                            {"role": "system", "content": "You are a biologist tasked with extracting biomarkers, endpoint scores, outcome measurement tools, and questionaire types."},
                                            {"role": "user", "content": mainOutcomes + " " + secondaryOutcomes}
                                        ],
                                        seed=1453,
                                        temperature=0)
    promptTokens += response1.usage.prompt_tokens
    completionTokens += response1.usage.completion_tokens

    response2 = openai.chat.completions.create(model="gpt-4",
                                        messages=[
                                            {"role": "system", "content": "You are a computer sciencetist tasked with creating a ontology from clinical trails in the OWL ontology code format."},
                                            {"role": "assistant", "content": "You have to convert the biomarkers, endpoint scores, outcome measurement tools, and questionaire types into an ontology."},
                                            {"role": "assistant", "content": "The only thing you have to do is put the biomarkers, endpoint scores, outcome measurement tools, and questionaires as subclasses of their respective mainclasses (i.e. ex:Biomarker, ex:EndpointScore, ex:MeasurementTool, and ex:Questionnaire)."},
                                            {"role": "assistant", "content": "Make sure to include the prefixes as shown in the example"},
                                            {"role": "assistant", "content": """Example: @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
                                                                                        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                                                                                        @prefix owl: <http://www.w3.org/2002/07/owl#> .
                                                                                        @prefix ex: <http://example.org/frailty#> .

                                                                                        # Main classes
                                                                                        ex:MeasurementTool a owl:Class .
                                                                                        ex:Questionnaire a owl:Class .
                                                                                        ex:Biomarker a owl:Class .
                                                                                        ex:EndpointScore a owl:Class .

                                                                                        # Subclasses for biomarkers
                                                                                        ex:Exhaustion rdfs:subClassOf ex:Biomarker .
                                                                                        ex:UnintentionalWeightLoss rdfs:subClassOf ex:Biomarker .
                                                                                        ex:PhysicalActivity rdfs:subClassOf ex:Biomarker .
                                                                                        ex:GripStrength rdfs:subClassOf ex:Biomarker .
                                                                                        ex:WalkingSpeed rdfs:subClassOf ex:Biomarker .

                                                                                        # Subclasses for measurement tools
                                                                                        ex:CESD rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:MinnesotaLeisureTimeActivityQuestionnaire rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:Dynamometer rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:BalanceTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:GaitSpeedTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:ChairStandsTest rdfs:subClassOf ex:MeasurementTool .

                                                                                        # Subclasses for endpoint scores
                                                                                        ex:HbA1cChange rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBBalanceScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBGaitSpeedScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBChairStandsScore rdfs:subClassOf ex:EndpointScore ."""},
                                            {"role": "user", "content": response1.choices[0].message.content},
                                        ],
                                        seed=1453,
                                        temperature=0)
    promptTokens += response2.usage.prompt_tokens
    completionTokens += response2.usage.completion_tokens

    return NCT, response2.choices[0].message.content, promptTokens, completionTokens

def chainedPromptGPT4o(clinicalTrail):
    NCT = str(clinicalTrail[0])
    mainOutcomes = clinicalTrail[1]
    secondaryOutcomes = clinicalTrail[2]
    promptTokens = 0
    completionTokens = 0

    response1 = openai.chat.completions.create(model="gpt-4o",
                                        messages=[
                                            {"role": "system", "content": "You are a biologist tasked with extracting biomarkers, endpoint scores, outcome measurement tools, and questionaire types."},
                                            {"role": "user", "content": mainOutcomes + " " + secondaryOutcomes}
                                        ],
                                        seed=1453,
                                        temperature=0)
    promptTokens += response1.usage.prompt_tokens
    completionTokens += response1.usage.completion_tokens

    response2 = openai.chat.completions.create(model="gpt-4o",
                                        messages=[
                                            {"role": "system", "content": "You are a computer sciencetist tasked with creating a ontology from clinical trails in the OWL ontology code format."},
                                            {"role": "assistant", "content": "You have to convert the biomarkers, endpoint scores, outcome measurement tools, and questionaire types into an ontology."},
                                            {"role": "assistant", "content": "The only thing you have to do is put the biomarkers, endpoint scores, outcome measurement tools, and questionaires as subclasses of their respective mainclasses (i.e. ex:Biomarker, ex:EndpointScore, ex:MeasurementTool, and ex:Questionnaire)."},
                                            {"role": "assistant", "content": "Make sure to include the prefixes as shown in the example"},
                                            {"role": "assistant", "content": """Example: @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
                                                                                        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
                                                                                        @prefix owl: <http://www.w3.org/2002/07/owl#> .
                                                                                        @prefix ex: <http://example.org/frailty#> .

                                                                                        # Main classes
                                                                                        ex:MeasurementTool a owl:Class .
                                                                                        ex:Questionnaire a owl:Class .
                                                                                        ex:Biomarker a owl:Class .
                                                                                        ex:EndpointScore a owl:Class .

                                                                                        # Subclasses for biomarkers
                                                                                        ex:Exhaustion rdfs:subClassOf ex:Biomarker .
                                                                                        ex:UnintentionalWeightLoss rdfs:subClassOf ex:Biomarker .
                                                                                        ex:PhysicalActivity rdfs:subClassOf ex:Biomarker .
                                                                                        ex:GripStrength rdfs:subClassOf ex:Biomarker .
                                                                                        ex:WalkingSpeed rdfs:subClassOf ex:Biomarker .

                                                                                        # Subclasses for measurement tools
                                                                                        ex:CESD rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:MinnesotaLeisureTimeActivityQuestionnaire rdfs:subClassOf ex:Questionnaire .
                                                                                        ex:Dynamometer rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:BalanceTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:GaitSpeedTest rdfs:subClassOf ex:MeasurementTool .
                                                                                        ex:ChairStandsTest rdfs:subClassOf ex:MeasurementTool .

                                                                                        # Subclasses for endpoint scores
                                                                                        ex:HbA1cChange rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBBalanceScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBGaitSpeedScore rdfs:subClassOf ex:EndpointScore .
                                                                                        ex:SPPBChairStandsScore rdfs:subClassOf ex:EndpointScore ."""},
                                            {"role": "user", "content": response1.choices[0].message.content},
                                        ],
                                        seed=1453,
                                        temperature=0)
    promptTokens += response2.usage.prompt_tokens
    completionTokens += response2.usage.completion_tokens

    return NCT, response2.choices[0].message.content, promptTokens, completionTokens

def cleanResponseContent2(responseContent):
    if responseContent.find("```") != -1:
        responseContent = responseContent.split("```")[1].strip()

    return responseContent[responseContent.find("@prefix"):]

def cleanResponseContent(responseContent):
    if "```" in responseContent:
        responseContent = responseContent.split("```")[1].strip()
    
    if "prefix" in responseContent:
        responseContent = responseContent[responseContent.find("@prefix"):]

    return responseContent

def saveClinicalTrailOntology(path, version, NCT, ontology):
    try:
        os.mkdir(str(path) + "v" + str(version))
    except Exception:
        pass

    file = open(str(path) + "v" + str(version) + "/" + NCT + ".owl", "w")
    file.write(ontology)
    file.close()

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


# timeLog("GPT3v4temp start")
# totalPromptTokens = 0
# totalCompletionTokens = 0
# totalTokens = 0
# totalPromptTokenCost = 0
# totalCompletionTokenCost = 0
# totalTokenCost = 0
# #import clinical trails from CSV
# clinicalTrails = getClinicalTrails()[1:51]

# #get either ontology as a whole or the information of interest (biomarkers, etc.) and create an ontology for it per clinical trail
# #Note to keep in mind prompt engineering, prompt chainging, function calling
# print("number of clinical trails: " + str(len(clinicalTrails)))
# for enum, clinicalTrail in enumerate(clinicalTrails):

#     NCT, responseContent, promptTokens, completionTokens = promptGPT3(clinicalTrail)

#     generatedOntology = cleanResponseContent(responseContent)
#     saveClinicalTrailOntology("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/ontologiesGPT3", "temp", NCT, generatedOntology)

#     tokenCounter(promptTokens, completionTokens)
#     costCounter(promptTokens, completionTokens, "gpt-3.5-turbo-1106")

#     print("Clinical trail: " + str(enum+1) + "/" + str(len(clinicalTrails)), "- " + str(NCT), "- prompt tokens: " + str(totalPromptTokens), "- completion tokens: " + str(totalCompletionTokens), "- total tokens: " + str(totalTokens), "- total cost: " + str(totalTokenCost))
#     rateLimitDelay(rpm=3500)
# timeLog("GPT3v4temp end")



# timeLog("chainedGPT3v4temp start")
# totalPromptTokens = 0
# totalCompletionTokens = 0
# totalTokens = 0
# totalPromptTokenCost = 0
# totalCompletionTokenCost = 0
# totalTokenCost = 0
# #import clinical trails from CSV
# clinicalTrails = getClinicalTrails()[1:51]

# #get either ontology as a whole or the information of interest (biomarkers, etc.) and create an ontology for it per clinical trail
# #Note to keep in mind prompt engineering, prompt chainging, function calling
# print("number of clinical trails: " + str(len(clinicalTrails)))
# for enum, clinicalTrail in enumerate(clinicalTrails):

#     NCT, responseContent, promptTokens, completionTokens = chainedPromptGPT3(clinicalTrail)

#     generatedOntology = cleanResponseContent(responseContent)
#     saveClinicalTrailOntology("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/chainedOntologiesGPT3", "temp", NCT, generatedOntology)

#     tokenCounter(promptTokens, completionTokens)
#     costCounter(promptTokens, completionTokens, "gpt-3.5-turbo-1106")

#     print("Clinical trail: " + str(enum+1) + "/" + str(len(clinicalTrails)), "- " + str(NCT), "- prompt tokens: " + str(totalPromptTokens), "- completion tokens: " + str(totalCompletionTokens), "- total tokens: " + str(totalTokens), "- total cost: " + str(totalTokenCost))
#     rateLimitDelay(rpm=3500)
# timeLog("chainedGPT3v4temp end")



# timeLog("GPT4v4temp start")
# totalPromptTokens = 0
# totalCompletionTokens = 0
# totalTokens = 0
# totalPromptTokenCost = 0
# totalCompletionTokenCost = 0
# totalTokenCost = 0
# #import clinical trails from CSV
# clinicalTrails = getClinicalTrails()[1:51]

# #get either ontology as a whole or the information of interest (biomarkers, etc.) and create an ontology for it per clinical trail
# #Note to keep in mind prompt engineering, prompt chainging, function calling
# print("number of clinical trails: " + str(len(clinicalTrails)))
# for enum, clinicalTrail in enumerate(clinicalTrails):

#     NCT, responseContent, promptTokens, completionTokens = promptGPT4(clinicalTrail)

#     generatedOntology = cleanResponseContent(responseContent)
#     saveClinicalTrailOntology("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/ontologiesGPT4", "temp", NCT, generatedOntology)

#     tokenCounter(promptTokens, completionTokens)
#     costCounter(promptTokens, completionTokens, "gpt-4")

#     print("Clinical trail: " + str(enum+1) + "/" + str(len(clinicalTrails)), "- " + str(NCT), "- prompt tokens: " + str(totalPromptTokens), "- completion tokens: " + str(totalCompletionTokens), "- total tokens: " + str(totalTokens), "- total cost: " + str(totalTokenCost))
#     rateLimitDelay(rpm=500)
# timeLog("GPT4v4temp end")


timeLog("chainedGPT4v5 start")
totalPromptTokens = 0
totalCompletionTokens = 0
totalTokens = 0
totalPromptTokenCost = 0
totalCompletionTokenCost = 0
totalTokenCost = 0
#import clinical trails from CSV
clinicalTrails = getClinicalTrails()[1:51]

#get either ontology as a whole or the information of interest (biomarkers, etc.) and create an ontology for it per clinical trail
#Note to keep in mind prompt engineering, prompt chainging, function calling
print("number of clinical trails: " + str(len(clinicalTrails)))
for enum, clinicalTrail in enumerate(clinicalTrails):
    try:
        NCT, responseContent, promptTokens, completionTokens = chainedPromptGPT4(clinicalTrail)

        generatedOntology = cleanResponseContent(responseContent)
        saveClinicalTrailOntology("/Users/berkancakir/Library/CloudStorage/OneDrive-VrijeUniversiteitAmsterdam/thesis/openAI/chainedOntologiesGPT4", "5", NCT, generatedOntology)

        tokenCounter(promptTokens, completionTokens)
        costCounter(promptTokens, completionTokens, "gpt-4")

        print("Clinical trail: " + str(enum+1) + "/" + str(len(clinicalTrails)), "- " + str(NCT), "- prompt tokens: " + str(totalPromptTokens), "- completion tokens: " + str(totalCompletionTokens), "- total tokens: " + str(totalTokens), "- total cost: " + str(totalTokenCost))
        rateLimitDelay(rpm=500)
    except Exception:
        pass
timeLog("chainedGPT4v5 end")