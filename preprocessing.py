import Sastrawi.Stemmer.StemmerFactory as StemmerFactory
import pickle


# load stopwords from txt file
with open('stopwords_id_satya.txt', 'r') as file:
    my_stopwords = file.read().splitlines()


# stemmer
factory = StemmerFactory.StemmerFactory()
stemmer = factory.create_stemmer()


def lowercase_and_remove_punctuation(text):
    text = text.lower()
    text = text.replace('[^\w\s]', '')
    return text

def remove_stopwords(text):
    text = text.split()
    text = [word for word in text if word not in my_stopwords]
    text = ' '.join(text)
    return text

def stemming(text):
    text = stemmer.stem(text)
    return text


def preprocess(text):
    text = lowercase_and_remove_punctuation(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

# main
if __name__ == "__main__":
    print(my_stopwords)