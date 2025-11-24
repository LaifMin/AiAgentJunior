import os
import yaml
import logging
import re
import asyncio
import time
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore



with open("config.yml", 'r') as ymlFiles:
    config = yaml.safe_load(ymlFiles)


app = Flask(__name__, template_folder=".")
logging.basicConfig(level = logging.INFO)

#loading the AI model
agentModel = ChatOllama(model = config['models']['ai'], temperature = 0.7, reasoning = False)

#Rag db 
embeddings = OllamaEmbeddings(model = config['models']['embed'])
vectorStoring = InMemoryVectorStore.load(config['rag_db_path'], embeddings)
retriever = vectorStoring.as_retriever()

#TTS not used now but goes here


#STT not used now but goes here

def is_prompt_safe(prompt):
   prompt = prompt.lower()
   message = config['prompts']['secure'].format(prompt = message)
   answer = agentModel.invoke([('human', message)])
   return not ("unsafe" in answer.lower())


context = [
    ('system', config['prompts']['default']),
    ('system', '')
]

context1 = [
    ('system', config['prompts']['giove']),
    ('system', '')
]

context2 = [
    ('system', config['prompts']['zeus']),
    ('system', '')
]

context3 = [
    ('system', config['prompts']['ade']),
    ('system', '')
]


contextDictionary = {
   'Giove': context1,
   'Zeus': context2,
   'Ade': context3
}


def get_contexts(character_name):
   contexts = []
   for name, char_ctx in contextDictionary.items():
      if name != character_name:
         conversation = char_ctx[2:]
         if conversation:
             contexts.append({
                    'character': name,
                    'messages': conversation
                })
   return contexts


def format_contexts(contexts):
   if not contexts:
      return ""

   formatted = "\n\n--- CONVERSAZIONI DEGLI ALTRI PERSONAGGI (per tua informazione) ---\n"
   for ctx in contexts:
      formatted += f"\n{ctx['character']}:\n"
      for role, content in ctx['messages']:
         if role == 'human':
            formatted += f"  Utente: {content}\n"
         elif role == 'ai':
            formatted += f"  {ctx['character']}: {content}\n"
   formatted += "\n--- FINE CONVERSAZIONI ALTRI PERSONAGGI ---\n"
   return formatted



def generate_answer(user_prompt):
   """
   if not is_prompt_safe(user_prompt):
      logging.warning("Unsafe message detected from user.")
      unsafeString = "Your message was detected as unsafe, thus I cannot answer it; please rephrase it or change the content."
      return {"answer": unsafeString
              }
   
   """
   logging.info("Generating answer for user prompt:" + user_prompt)
   
   context.append(('human', user_prompt))

   #Retrieve relevant documents from RAG
   relevantDocs = retriever.invoke(user_prompt)
   doc_text = "\n".join([d.page_content for d in relevantDocs])
   doc_text = "Usa questo testo per rispondere alla domanda:\n" + doc_text + "\nSe non conosci la risposta, dì che non lo sai.\n"
   context[1] = ('system', doc_text)

   answer = agentModel.invoke(context).content
   context.append(('ai', answer))
   return answer

def generate_answerMC(user_prompt, character_name):
   """
   if not is_prompt_safe(user_prompt):
      logging.warning("Unsafe message detected from user.")
      unsafeString = "Your message was detected as unsafe, thus I cannot answer it; please rephrase it or change the content."
      return {"answer": unsafeString
              }
   
   """
   logging.info("Generating answer for user prompt:" + user_prompt)
   context = contextDictionary[character_name]
   context.append(('human', user_prompt))

   #Retrieve relevant documents from RAG
   relevantDocs = retriever.invoke(user_prompt)
   doc_text = "\n".join([d.page_content for d in relevantDocs])
   doc_text = "Usa questo testo per rispondere alla domanda:\n" + doc_text + "\nSe non conosci la risposta, dì che non lo sai.\n"
   
   
   other_contexts = get_contexts(character_name)
   other_contexts_text = format_contexts(other_contexts)
   doc_text += other_contexts_text
    
   context[1] = ('system', doc_text)

   answer = agentModel.invoke(context).content
   context.append(('ai', answer))
   return answer




@app.route('/')
def index():
   return render_template('index.html', history = context)

@app.route('/question', methods=['POST'])
def question():
   userPrompt = request.get_json()
   answer = userPrompt.get('question', 'No question')
   responseText = generate_answer(answer)
   return jsonify({'answer': responseText,
                   'ai': 'yes'
                   })

@app.route('/giove', methods=['POST'])
def giove():
   userPrompt = request.get_json()
   answer = userPrompt.get('question', 'No question')
   responseText = generate_answerMC(answer, 'Giove')
   return jsonify({'answer': responseText,
                   'char': 'giove'
                   })


@app.route('/zeus', methods=['POST'])
def zeus():
   userPrompt = request.get_json()
   answer = userPrompt.get('question', 'No question')
   responseText = generate_answerMC(answer, 'Zeus')
   return jsonify({'answer': responseText,
                   'char': 'zeus'
                   })

@app.route('/ade', methods=['POST'])
def ade():
   userPrompt = request.get_json()
   answer = userPrompt.get('question', 'No question')
   responseText = generate_answerMC(answer, 'Ade')
   return jsonify({'answer': responseText,
                   'char': 'ade'
                   })

CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

if __name__ == "__main__":
   app.run(host='0.0.0.0', port=config['server_portc'], debug=True)

