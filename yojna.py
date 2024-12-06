from flask import Flask, render_template, request, redirect, url_for,jsonify
import os
import speech_recognition as sr
import time
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from deep_translator import GoogleTranslator
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain    
from langchain_community.document_loaders import PyPDFLoader
from playsound import playsound


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'ogg', 'flac','m4a'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('Listening') 
            r.pause_threshold = 0.7 
            time.sleep(2)
            audio_text = r.listen(source)
            print("Time over, thanks")
            text = r.recognize_google(audio_text, language='hi-IN')
            print(text)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], "microphone-results.wav")
            audio_file= open(filename, "rb")
            print(audio_file)
            with open(filename, "wb") as file:
              file.write(audio_text.get_wav_data())
            filenaame = os.path.join(app.config['UPLOAD_FOLDER'], "microphone-results.wav")
            client = OpenAI()
            transcription = client.audio.translations.create(
            model="whisper-1", 
            file=audio_file
            )
            print(transcription.text)
            return redirect(url_for('uploaded_file',text = text))
    return render_template('index.html')
@app.route('/uploads/')
def uploaded_file():
    client = OpenAI()
    filename = os.path.join(app.config['UPLOAD_FOLDER'], "microphone-results.wav")
    audio_file= open(filename, "rb")
    transcription = client.audio.translations.create(
    model="whisper-1", 
    file=audio_file,)
    loader = PyPDFLoader('kusumyojna.pdf')
    docs = loader.load()
    prompt = ChatPromptTemplate.from_template("""Answer the following
    <context>
    {context}
    </context>

    Question:{input}
    """)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    chunk_documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = FAISS.from_documents(chunk_documents,embeddings)
    query = transcription.text
    translated_query = GoogleTranslator(source='auto', target='hi').translate(query)
    print(translated_query)  
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({"input":query})
    answer = response['answer']
    translated_response = GoogleTranslator(source='auto', target='hi').translate(answer)
    print(translated_response)  
    from pathlib import Path
    speech_file_path = Path(__file__).parent / "speech.mp3"
    print(speech_file_path)
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input = translated_response,
        
    )
    response.stream_to_file(speech_file_path)
    print(response)
    from playsound import playsound
    playsound('speech.mp3')
    os.remove('speech.mp3')
    return render_template("Index.html",answer = translated_response,query=translated_query)

if __name__ == '__main__':
    app.run(debug=True)
