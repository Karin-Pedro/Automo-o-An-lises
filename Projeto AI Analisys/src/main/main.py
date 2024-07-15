from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import text
from util import leitor_excel

simulacao = leitor_excel.ler_planilha('simulacao_dados.xlsx')

texts = simulacao['DADOS'].to_list()
#
# # Carregar stop words padrão e adicionar termos específicos ao contexto
custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(['foi', 'da', 'do', "de", "mas", "das", "dos", "tem", "uma"]))
#
# # CLUSTERIZAÇÃO - RÓTULOS DINÂMICOS
vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
x = vectorizer.fit_transform(texts)
#
kmeans = KMeans(n_clusters=10)
kmeans.fit(x)
#
# Gerar termos principais de cada cluster
cluster_terms = []
for i in range(kmeans.n_clusters):  # Número de clusters
    cluster_center = kmeans.cluster_centers_[i]
    top_terms_indices = cluster_center.argsort()[-3:]  # Top 3 termos
    top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_terms_indices]
    cluster_terms.append(" ".join(top_terms))

# Mapeando textos aos clusters
labels = kmeans.labels_

# UTILIZANDO spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)


preprocessed_texts = [preprocess(text) for text in texts]

# Criar um pipeline de classificação usando scikit-learn
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Treinar o modelo usando textos pré-processados e seus respectivos rótulos de cluster
model.fit(preprocessed_texts, labels)


def processar_colunas(df):
    for coluna in df.columns:
        new_texts = df[coluna].to_list()
        for text in new_texts:
            preprocessed_new_text = preprocess(text)
            predicted_label = model.predict([preprocessed_new_text])[0]
            predicted_cluster_terms = cluster_terms[predicted_label]
            print(f'Texto: "{text}"\nRótulo: {predicted_label}\nTermos do cluster: {predicted_cluster_terms}\n\n')


feedbacks = leitor_excel.ler_planilha('first_feedbacks_for_test.xlsx')
df_filtrado = leitor_excel.remover_colunas_escala(feedbacks)
processar_colunas(df_filtrado)