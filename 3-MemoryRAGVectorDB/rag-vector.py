#Embedding : transform the text into numerical vectors that hold semantic distance
#Vector DB : store the vectors and compare them

from template import AGENT_ROLE_TEMPLATE, use, populate, prompt, init_agent, llm_config

# ---STRING DOCUMENTS---
DOC1 = '''
Cool documentation page 1
Greeting function
Greets another user with a message like so :
greetings(name1, name2, message) => "<name1> says : "Hi <name2> ! <message>"
'''

DOC2 = '''
Cool documentation page 2
Anything function
Gives a random generated message about a different subject every time you use it.
Useful to get random text to test your formatting or research algorithms.
anything() => <random_message>
'''

# ---PROMPT---
PROMPT = str(input("Write a short prompt here : \n"))

# Add docs and query db
import chromadb

# ---CHROMA CLIENT---
# Use PersistentClient to save data to a local folder
client = chromadb.PersistentClient(path="./test-rag")

# ---CREATE COLLECTION AND ADD DOCUMENTS---
# A collection is like a table in a database
collection = client.get_or_create_collection(name="test-docs")

# Add text to the collection
# Chroma automatically generates embeddings if you don't provide them
collection.add(
    documents=[DOC1, DOC2],
    metadatas=[{"source": "cool-documentation"}, {"source": "cool-documentation"}],
    ids=["id1", "id2"]
)

# ---QUERY DATABASE---
results = collection.query(
    query_texts=[PROMPT],
    n_results=1
)

# ---PARSE RESULTS---
# parse results into a list of dicts to use in the agent
def parse(results):
    parsed_results = []
    
    # results['documents'][0] and results['metadatas'][0] are lists
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        # Construct the dictionary with the mandatory 'type' key
        element = {
            "type": "text",  # This is the key your error is looking for
            "text": f"Source: {meta.get('source', 'Unknown')} \nContent: {doc}"
        }
        parsed_results.append(element)
        
    return parsed_results

#print(results)
results = parse(results)
use(AGENT_ROLE_TEMPLATE,results)



