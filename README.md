## Commands to run on production

RAG_TTS_Ollama_Edge.py is being used by api.py on a server to run continuously as an api endpoint.

The process is running on port 8001, can be accessed via an ngrok tunnel:
> TBD

#### Command to run RAG api:
After activating the venv on the server (that uses packages in requirements-app.txt)
uvicorn api:app --host 0.0.0.0 --port 8001

#### Command to run TTS api:
After activating the venv on the server (that uses packages in requirements-app.txt)
TBD

#### Command on server to run ngrok:
##### RAG:
TBD
##### TTS: 
TBD

(in case of auth issues:
ngrok config add-authtoken yourauthtoken)


## Deployment info
### Screen info
Fastapi and ngrok can run in a screen process (two windows)
- Attach to the screen:
`screen -r <screen_name>`
- Switch windows:
`Ctrl + A`, then `N`
- Detach from screen
`Ctrl + A`, then `D`

### Deployed API
Access api here:
TBD

To talk to the llm:
TBD/llm?input=Hello how are you
(Use the query parameter input to provide user input)
