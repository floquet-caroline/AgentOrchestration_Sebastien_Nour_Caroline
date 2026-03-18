# AgentOrchestration_Sebastien_Nour_Caroline


## Caroline - Simple RAG

For this part you'll need a venv with autogen, dotenv and chromadb installed and a Groq API key.

### 1. Simple prompt

This subpart just allows you to prompt the raw LLM, without anything else attached.

### 2. Prompt template

This subpart adds templating to the simple prompt, giving a specific context to the answer. You can change the template and replace it with your own to accomodate your needs.

### 3. Memory RAG and vector database

This last subpart allows you to query a simple database and retrieve data in a legible form thanks to the LLM. 

It uses both 1 and 2 attached to a simple vector database with basic embeddings. You can fill that database with pretty much anything and query it with a prompt. Just make sure to check the template is correctly aligned with your context for better results.