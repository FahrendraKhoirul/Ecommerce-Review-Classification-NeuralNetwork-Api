import numpy as np

class TFIDF:
    def __init__(self, data):
        self.data = data
        self.word_list = self.create_word_list()
        self.word_count_list = self.create_word_count_list()
        self.tf_matrix = self.create_tf_matrix()
        self.idf_vector = self.create_idf_vector()
        self.tfidf_matrix = self.create_tfidf_matrix()


    def create_word_list(self):
        word_list = []
        for sentence in self.data:
            for word in sentence.split():
                if word not in word_list:
                    word_list.append(word)
        return word_list
    
    def create_word_count_list(self):  # DF
        word_count = {}
        for w in self.word_list:
            word_count[w] = 0
            for sentence in self.data:
                for word in sentence.split():
                    if word == w:
                        word_count[w] += 1
        return word_count
    
    def count_tf(self, sentence):
        tf_vector = [0] * len(self.word_list) # output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for word in sentence.split():
            for i, w in enumerate(self.word_list): # i = index, w = word
                if w == word:
                    tf_vector[i] += 1
        # divide each element in tf_vector by the length of the sentence
        tf_vector = np.array(tf_vector) / len(sentence.split())
        return tf_vector
    

    def create_tf_matrix(self):
        tf_matrix = []
        for sentence in self.data:
            tf_matrix.append(self.count_tf(sentence))
        return tf_matrix
    
    def create_idf_vector(self): #
        idf_vector = []
        length_data = len(self.data)
        for w in self.word_list:
            count  = 0
            for sentence in self.data:
                if w in sentence.split():
                    count += 1
            idf_vector.append(np.log(length_data / count))
        return idf_vector
    
    def create_tfidf_matrix(self):
        tfidf_matrix = []
        for tf_vector in self.tf_matrix:
            tfidf_matrix.append(np.multiply(tf_vector, self.idf_vector))
        return tfidf_matrix

    def transform(self, sentence):
        tf_vector = self.count_tf(sentence)
        tfidf_vector = np.multiply(tf_vector, self.idf_vector)
        return tfidf_vector
    


if __name__ == "__main__":
    data = ["hello hello down there", "hello up there", "hello down there asd apa iya ahha", "hello up there"]
    tfidf = TFIDF(data)
    print(tfidf.transform("hello up there"))

