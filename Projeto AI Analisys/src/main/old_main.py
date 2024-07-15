from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import text

# SIMULAÇÃO DE DADOS
texts = [
    "Achei muito boa didática do professor",
    "Gostei muito da aula, o professor explica muito bem",
    "Fiquei focado na aula com a didática do professor",
    "Foi bem fácil de entender as explicações",
    "Aula bastante produtiva, entendi tudo",
    "Ótima aula",
    "Professor muito bom",
    "A didática do professor é excelente",
    "As explicações foram claras e objetivas",
    "Aula dinâmica e envolvente",
    "Gostei da interação durante a aula",
    "Professor soube tirar todas as dúvidas",
    "Conteúdo foi bem apresentado",
    "Aula prática e teórica muito bem equilibrada",
    "Explicações detalhadas e compreensíveis",
    "Aula com muitos exemplos práticos",
    "Achei o ritmo da aula muito bom",
    "Gostei da abordagem utilizada pelo professor",
    "Aula foi esclarecedora e informativa",
    "Professor muito atencioso",
    "Aula bastante enriquecedora",
    "Didática do professor ajuda muito na compreensão",
    "Conteúdo relevante e bem explicado",
    "Aula agradável e produtiva",
    "Gostei da metodologia aplicada",
    "Professor domina o assunto",
    "Aula bastante interativa",
    "Explicações claras e concisas",
    "Achei a aula um pouco rápida, mas boa",
    "Professor fez a aula ficar interessante",
    "Gostei das atividades propostas na aula",
    "Aula foi bem organizada",
    "Professor utiliza exemplos práticos que facilitam o entendimento",
    "Aula foi bastante motivadora",
    "Gostei do material de apoio fornecido",
    "Professor tem uma boa comunicação",
    "Aula foi bastante produtiva",
    "Didática do professor é muito boa",
    "Aula foi muito elucidativa",
    "Professor respondeu todas as perguntas com paciência",
    "Gostei da estrutura da aula",
    "Aula foi divertida e informativa",
    "Professor usa uma linguagem simples e direta",
    "Gostei dos recursos visuais utilizados na aula",
    "Aula com conteúdo bem aprofundado",
    "Professor consegue manter a turma engajada",
    "Aula foi muito proveitosa",
    "Gostei do ritmo da aula",
    "Professor tem uma boa didática",
    "Aula foi esclarecedora e motivadora"
]

# Carregar stop words padrão e adicionar termos específicos ao contexto
custom_stop_words = list(text.ENGLISH_STOP_WORDS.union(['foi', 'da', 'do', "de","mas", "das", "dos", "tem", "uma"]))

# CLUSTERIZAÇÃO - RÓTULOS DINÂMICOS
vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
x = vectorizer.fit_transform(texts)

kmeans = KMeans(n_clusters=10)
kmeans.fit(x)

# Gerar termos principais de cada cluster
cluster_terms = []
for i in range(kmeans.n_clusters):  # Número de clusters
    cluster_center = kmeans.cluster_centers_[i]
    top_terms_indices = cluster_center.argsort()[-3:]  # Top 3 termos
    top_terms = [vectorizer.get_feature_names_out()[idx] for idx in top_terms_indices]
    cluster_terms.append(" ".join(top_terms))
print("Cluster terms:\n", cluster_terms)

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

# Exemplo de classificação
new_texts = [
    "A aula foi excelente, muito esclarecedora",
    "Gostei bastante da forma como o professor explica",
    "A didática do professor é muito boa",
    "Aula bem organizada e fácil de acompanhar",
    "O conteúdo foi passado de maneira clara",
    "A interação com os alunos foi ótima",
    "Professor conseguiu manter a turma interessada",
    "Aula bem dinâmica e envolvente",
    "Gostei do uso de exemplos práticos",
    "Professor domina bem o assunto",
    "Aula bastante produtiva",
    "Explicações foram diretas e objetivas",
    "Gostei do ritmo da aula",
    "A metodologia utilizada foi excelente",
    "Professor respondeu todas as dúvidas",
    "Aula com bastante conteúdo relevante",
    "Gostei da forma como a aula foi conduzida",
    "Aula interativa e interessante",
    "Professor tem uma boa comunicação",
    "Aula foi bastante esclarecedora",
    "Gostei das atividades propostas",
    "Aula foi motivadora e inspiradora",
    "Professor utiliza bons exemplos",
    "A didática do professor facilita o aprendizado",
    "Aula foi bem estruturada",
    "Gostei do material de apoio",
    "Aula foi bastante proveitosa",
    "Explicações foram claras e detalhadas",
    "Professor soube engajar a turma",
    "Aula prática e teórica bem balanceada",
    "Gostei da abordagem do professor",
    "Aula foi interessante e produtiva",
    "Professor consegue transmitir bem o conteúdo",
    "Aula foi bastante interativa",
    "Gostei da clareza das explicações",
    "Aula foi bem dinâmica",
    "Professor tem uma ótima didática",
    "Aula foi informativa e envolvente",
    "Gostei do uso de recursos visuais",
    "Aula com conteúdo bem aprofundado",
    "Professor tem uma boa comunicação",
    "Aula foi agradável e produtiva",
    "Gostei da estrutura da aula",
    "Aula foi esclarecedora e interessante",
    "Professor conseguiu manter a turma atenta",
    "Aula com muitos exemplos práticos",
    "Gostei do ritmo e da dinâmica da aula",
    "Professor soube transmitir o conteúdo de forma clara",
    "Aula foi motivadora e esclarecedora",
    "Gostei da forma como o conteúdo foi apresentado",
    "Aula foi bastante proveitosa e esclarecedora"
]
for text in new_texts:
    preprocessed_new_text = preprocess(text)
    predicted_label = model.predict([preprocessed_new_text])[0]
    predicted_cluster_terms = cluster_terms[predicted_label]
    print(f'Texto: "{text}"\nRótulo: {predicted_label}\nTermos do cluster: {predicted_cluster_terms}\n\n')

