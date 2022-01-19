import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize

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

