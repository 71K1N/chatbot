import pygame
import speech_recognition as sr
from gtts import gTTS

# ----------------------------------------------------------------
# FUNÇÃO PARA OUVIR E RECONHECER A FALA
# ----------------------------------------------------------------
def ouvir_microfone():
    # Habilita o microfone do usuário
    microfone = sr.Recognizer()

    # usando o microfone
    with sr.Microphone() as source:

        # Chama um algoritmo de reducao de ruidos no som
        microfone.adjust_for_ambient_noise(source)

        # Armazena o que foi dito numa variavel
        audio = microfone.listen(source)

    try:
        # Passa a variável para o algoritmo reconhecedor de padroes
        frase = microfone.recognize_google(audio, language='pt-br')

        # Retorna a frase pronunciada
        print("Você disse: " + frase)

    # Se nao reconheceu o padrao de fala, exibe a mensagem
    except sr.UnknownValueError:
        frase = "Não entendi"

    return frase

# ----------------------------------------------------------------
# CRIA ARQUIVO DE AUDIO 
# ----------------------------------------------------------------
def cria_audio(audio):
    tts = gTTS(audio, lang='pt-br')    
    tts.save('hello.mp3')

# ----------------------------------------------------------------
# EXECUTA O AUDIO
# ----------------------------------------------------------------
def playAudio():    
    pygame.mixer.init()    
    pygame.mixer.music.load("hello.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue
    
