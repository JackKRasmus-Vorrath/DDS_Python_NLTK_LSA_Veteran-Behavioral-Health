#%pwd
#%cd Desktop

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#processes XML file (record of patient info wrapped in tags)

info = open('Veterans_Behavioral_Health.xml', 'r+').read()

soup = BeautifulSoup(info, 'lxml')		#lxml parser
conditionTxt = soup.findAll('measure_name')
conditionDocs = [x.text for x in conditionTxt]
#conditionDocs.pop(0)
conditionDocs = [x.lower() for x in conditionDocs]

stopset = set(stopwords.words('english'))
stopset.update(['with', 'on', 'to', 'were', 'appropriate', 'justification', ])

conditionTxt[0] 

vectorizer = TfidfVectorizer(stop_words = stopset, use_idf = True, ngram_range = (1, 3))
X = vectorizer.fit_transform(conditionDocs)

X[0]

print(X[0])

# X: matrix where m is # docs, n # terms

# X = USV^(t)

#decomposition of three matrices U, S, T, picking a value k (# of concepts to keep)

#U: m x k matrix (docs x concepts)
#S: k x k diagonal matrix (elements = amount of variation from each concept)
#V: m x k (transposed) matrix (terms x concepts)

X.shape
#(documents x terms)

lsa = TruncatedSVD(n_components = 5, n_iter = 100)		#singular value decomposition
lsa.fit(X)

#first row for V: all the terms that go with row [0]; how important the term is to that concept, with position in the row relating to the position of the term

lsa.components_[0]

terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_):
	termsInComp = zip(terms, comp)
	sortedTerms = sorted(termsInComp, key = lambda x:x[1], reverse = True)[:6]		#6 terms per 5 concepts
	print("Concept %d:" % i)
	for term in sortedTerms:
		print(term[0])
	print(" ")
	






