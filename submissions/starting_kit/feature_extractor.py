from __future__ import unicode_literals
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata
import pandas as pd
import numpy as np

def document_preprocessor(doc):
    # TODO: is there a way to avoid these encode/decode calls?
    try:
        doc = unicode(doc, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    doc = unicodedata.normalize('NFD', doc)
    doc = doc.encode('ascii', 'ignore')
    doc = doc.decode("utf-8")
    return str(doc)


from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

def token_processor(tokens):
    for token in tokens:
        #remove special chars
        token=''.join(e for e in token if e.isalnum())
        yield stemmer.stem(token)

class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):
        # see ``TfidfVectorizer`` documentation for other feature
        # extraction parameters.
        super(FeatureExtractor, self).__init__(
                analyzer='word',stop_words ='english', preprocessor=document_preprocessor)

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``statement``
            column.
        """
        
        super(FeatureExtractor, self).fit(X_df.TEXT)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        X = super(FeatureExtractor, self).transform(X_df.TEXT)
        #ids = np.array(X_df['HADM_ID'].reshape((X.shape[0], 1)) )
        #X_ = np.concatenate((ids, X.todense()),axis=1)
        return X

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))
