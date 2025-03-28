## Commands to run on production
RAG_TTS_Ollama_Edge.py is being used by api.py on a server to run continuously as an api endpoint.

#### Command to run RAG api:
After activating the venv on the server (that uses packages in requirements-app.txt) <br />
`uvicorn api:app --host 0.0.0.0 --port 8001`

#### Command to run TTS api:
After activating the venv on the server (that uses packages in requirements-app.txt) <br />
`uvicorn edgetts:app --host 0.0.0.0 --port 8000`

#### Command on server to run ngrok:
##### RAG:
`ngrok http --url=fancy-bengal-uniformly.ngrok-free.app 8001`
##### TTS: 
`ngrok http --url=crane-pure-closely.ngrok-free.app 8000`<br />

(in case of auth issues:
`ngrok config add-authtoken <yourauthtoken>`)

## Deployment info
### Screen info
Fastapi and ngrok can run in a screen process (two windows)
- Attach to the screen:
`screen -r <screen_name>`
- Switch windows:
`Ctrl + A`, then `N`
- Detach from screen
`Ctrl + A`, then `D` <br />


To talk to the RAG: <br />
https://fancy-bengal-uniformly.ngrok-free.app/llm?input="Hello how are you?" <br />

To listen to the TTS: <br />
https://crane-pure-closely.ngrok-free.app/tts/?text="Hello, can you hear me?"
