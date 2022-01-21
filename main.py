import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import unidecode

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk import tokenize, ngrams
from string import punctuation

#download imdb dataset :https://www.kaggle.com/luisfredgs/imdb-ptbr

def classificar_texto(texto, coluna_texto, coluna_classificacao):
    vetorizar = CountVectorizer(lowercase=False, max_features=50)

    bag_of_words = vetorizar.fit_transform(texto[coluna_texto])

    # vetorizar.get_feature_names()

    # matriz = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names())

    # print(bag_of_words.shape)

    treino, teste, classe_treino, classe_teste = train_test_split(bag_of_words,
                                                                texto[coluna_classificacao],
                                                                random_state=42)

    regressao_logistica = LogisticRegression(solver='lbfgs')

    regressao_logistica.fit(treino,classe_treino)

    return regressao_logistica.score(teste,classe_teste)

def pareto(texto, coluna_texto,quantidade):
    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])

    token_espaco = tokenize.WhitespaceTokenizer()

    token_frase = token_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)

    df_frequencia = pd.DataFrame({'Palavras' : list(frequencia.keys()),
                                'Frequencia': list(frequencia.values())}).nlargest(columns='Frequencia',n=quantidade)

    plt.figure(figsize=(12,8))

    ax = sns.barplot(data = df_frequencia,x = 'Palavras', y = 'Frequencia', color='gray')
    ax.set(ylabel='contagem')
    plt.show


resenha = pd.read_csv('imdb-reviews-pt-br.csv')

resenha['classificacao'] = resenha['sentiment'].replace(['neg','pos'],[0,1])

print(classificar_texto(resenha,'text_pt','classificacao'))

# nltk.download('all') # if first time

pareto(resenha,'text_pt',10)

palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')

frase_processada = list()
for opiniao in resenha.text_pt:
    nova_frase = list()
    palavras_texto  = tokenize.WhitespaceTokenizer().tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in palavras_irrelevantes:
            nova_frase.append(palavra)

    frase_processada.append(' '.join(nova_frase))

resenha['tratamento_1'] = frase_processada

classificar_texto(resenha,'tratamento_1','classificacao')

pareto(resenha, 'tratamento_1',10)

token_pontuacao = tokenize.WordPunctTokenizer()
pontuacao = list()
for ponto in punctuation:
    pontuacao.append(ponto)

pontuacao_stopwords = pontuacao + palavras_irrelevantes

frase_processada = list()
for opiniao in resenha["tratamento_1"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha['tratamento_2'] = frase_processada

pareto(resenha,'tratamento_2',10)

sem_acentos = [unidecode.unidecode(texto) for texto in resenha['tratamento_2']]

stopwords_sem_acento = [unidecode.unidecode(texto) for texto in pontuacao_stopwords]

resenha['tratamento_3'] = sem_acentos

frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in pontuacao_stopwords:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha['tratamento_3'] = frase_processada

acuracia_tratamento3 = classificar_texto(resenha,'tratamento_3','classificacao')

pareto(resenha, 'tratamento_3',10)

frase_processada = list()
for opiniao in resenha["tratamento_3"]:
    nova_frase = list()
    opiniao = opiniao.lower()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(palavra)
    frase_processada.append(' '.join(nova_frase))

resenha['tratamento_4'] = frase_processada
acuracia_tratamento4 = classificar_texto(resenha,'tratamento_4','classificacao')

print('acuracia 4 :',acuracia_tratamento4)
pareto(resenha, 'tratamento_4',10)

stemmer = nltk.RSLPStemmer()

frase_processada = list()
for opiniao in resenha["tratamento_4"]:
    nova_frase = list()
    palavras_texto = token_pontuacao.tokenize(opiniao)
    for palavra in palavras_texto:
        if palavra not in stopwords_sem_acento:
            nova_frase.append(stemmer.stem(palavra))
    frase_processada.append(' '.join(nova_frase))

resenha["tratamento_5"] = frase_processada

acuracia_tratamento5 = classificar_texto(resenha, "tratamento_5", "classificacao")
print('acuracia 5 :',acuracia_tratamento5)
pareto(resenha, "tratamento_5", 10)

# tfidf = TfidfVectorizer(lowercase=False, max_features=50)

# tfidf_dados = tfidf.fit_transform(resenha['tratamento_5'])

# treino, teste, classe_treino, classe_teste = train_test_split(tfidf_dados,
#                                                               resenha['classificacao'],
#                                                               random_state=42)

# regressao_logistica = LogisticRegression(solver='lbfgs')

# regressao_logistica.fit(treino, classe_treino)

# acuracia_tratamento6 = regressao_logistica.score(teste,classe_teste)
# print('acuracia tfidf :',acuracia_tratamento6)

regressao_logistica = LogisticRegression(solver='lbfgs')

tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))
vetor_tfidf = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, resenha["classificacao"], random_state = 42)

regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)
print('acuracia ngrams: ',acuracia_tfidf_ngrams)

tfidf = TfidfVectorizer(lowercase=False)
vetor_tfidf = tfidf.fit_transform(resenha["tratamento_5"])
treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf, resenha["classificacao"], random_state = 42)
regressao_logistica.fit(treino, classe_treino)
acuracia_tfidf = regressao_logistica.score(teste, classe_teste)
print('acuracia sem ngrams: ',acuracia_tfidf)

pesos = pd.DataFrame(
    regressao_logistica.coef_[0].T,
    index = tfidf.get_feature_names()
)
pesos.nlargest(10, 0)

pesos.nsmallest(10,0)


