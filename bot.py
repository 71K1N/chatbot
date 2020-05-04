import nltk
import random
import json as js
import numpy as np
import tflearn as tfl
import tensorflow as tf
import tratamentoVoz as voz
from nltk.stem.rslp import RSLPStemmer

nltk.download('stopwords')
# ----------------------------------------------------------------
# FUNCAO REMOVER STOP WORDS
# ----------------------------------------------------------------


def RemoveStopWords(sentence):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    phrase = []
    for word in sentence:
        if word not in stopwords:
            phrase.append(word)
    return phrase


# IMPORTANDO ARQUIVO
with open("intents.json") as file:
    data = js.load(file)

# PREPARANDO DADOS
nltk.download('rslp')
# nltk.download('punkt')

palavras = []
intencoes = []
sentencas = []
saidas = []

for intent in data["intents"]:
    tag = intent['tag']
    if tag not in intencoes:
        intencoes.append(tag)
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern, language='portuguese')
        palavras.extend(wrds)
        sentencas.append(wrds)
        saidas.append(tag)

# REMOVENDO STOPWORDS
palavras = RemoveStopWords(palavras)

# STEMING
stemer = RSLPStemmer()
stemmed_words = [stemer.stem(w.lower()) for w in palavras]
stemmed_words = sorted(list(set(stemmed_words)))

# BAG OF WORDS
training = []
output = []
# criando um array preenchido com 0
outputEmpty = [0 for _ in range(len(intencoes))]

for x, frase in enumerate(sentencas):
    bag = []
    wds = [stemer.stem(k.lower()) for k in frase]
    for w in stemmed_words:
        if w in wds:
            bag.append(1)
        else:
            bag.append(0)

    outputRow = outputEmpty[:]
    outputRow[intencoes.index(saidas[x])] = 1

    training.append(bag)
    output.append(outputRow)

# REDE NEURAL
training = np.array(training)
output = np.array(output)

# reiniciando os dados
tf.reset_default_graph()

# camada de entrada
net = tfl.input_data(shape=[None, len(training[0])])
# oito neuronios por camada oculta
net = tfl.fully_connected(net, 8)
# camada de saida
net = tfl.fully_connected(net, len(output[0]), activation="softmax")
#
net = tfl.regression(net)
# criando o modelo
model = tfl.DNN(net)

# TREINANDO O MODELO
model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)
model.save("model.chatbot30G")  # ARQUIVO DE SAIDA

# ----------------------------------------------------------------
# BAG OF WORDS
# ----------------------------------------------------------------


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# ----------------------------------------------------------------
# VERIFICAR CERTEZA (entrada nao reconhecida)
# ----------------------------------------------------------------


def verificarCerteza(probabilidades):
    maximo = probabilidades.max()
    if maximo < 0.30:
        return False
    else:
        return True

# ----------------------------------------------------------------
# CHAT
# ----------------------------------------------------------------


def chat():

    print("Oi sou a Ayla! Converse com comigo !!! ")
    Online = True
    while Online:
        print("Sua vez: ")
        inp = voz.ouvir_microfone()

        bag_usuario = bag_of_words(inp, stemmed_words)
        results = model.predict([bag_usuario])
        # retorna se o maior indice de similaridade e aceito
        reconhecido = verificarCerteza(results)

        # VERIFICA SE O BOT TEM CERTEZA
        if(reconhecido == False):
            voz.cria_audio('Não intendi, pergunte outra vez')
            voz.playAudio()
        else:
            results_index = np.argmax(results)
            tag = intencoes[results_index]

            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            voz.cria_audio(random.choice(responses))
            voz.playAudio()

            if tag == "tchau":
                Online = False


chat()
