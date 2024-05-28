import numpy as np
import joblib
import re
import sys
import time
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import nltk
from nltk.corpus import words
from metaphone import doublemetaphone
import time

# nltk.download('words')
# correct_words = set(words.words())

# Metaphone fonksiyonu
def correct_word_metaphone(word,correct_words):
    word = word.lower()
    # Metaphone kodunu hesapla
    metaphone_code = doublemetaphone(word)[0]

    # Doğru kelimeyi bul
    best_match = None
    best_similarity = float('inf')
    for correct_word in correct_words:
        if doublemetaphone(correct_word)[0] == metaphone_code:
            similarity = nltk.edit_distance(word, correct_word)
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = correct_word
    if(best_match == None):
        return "No Solution"
    return best_match


# Damerau Levenshtein Distance fonksiyonu
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)

    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            cost = 0 if s1[i] == s2[j] else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,  # deletion
                d[(i, j - 1)] + 1,  # insertion
                d[(i - 1, j - 1)] + cost,  # substitution
            )
            if i > 0 and j > 0 and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]

def correct_word_damerau_levenshtein(word, word_list, max_distance):
    suggestions = []
    for w in word_list:
        distance = damerau_levenshtein_distance(word, w)
        if distance <= max_distance:
            suggestions.append((w, distance))
    suggestions.sort(key=lambda x: x[1])
    return suggestions

# Levenshtein Distance fonksiyonu
def levenshtein_distance(word1, word2):
    if len(word1) < len(word2):
        return levenshtein_distance(word2, word1)
    
    if len(word2) == 0:
        return len(word1)
    
    previous_row = range(len(word2) + 1)
    for i, c1 in enumerate(word1):
        current_row = [i + 1]
        for j, c2 in enumerate(word2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def correct_word_levenshtein(word, dictionary):
    closest_word = min(dictionary, key=lambda x: levenshtein_distance(word, x))
    return closest_word

# BK-Tree yapısı
class BKTree:
    def __init__(self, distance_func):
        self.tree = None
        self.distance_func = distance_func

    def add(self, word):
        node = BKTreeNode(word)
        if self.tree is None:
            self.tree = node
        else:
            self.tree.add(node, self.distance_func)

    def search(self, word, max_distance):
        if self.tree is None:
            return []
        return self.tree.search(word, max_distance, self.distance_func)

class BKTreeNode:
    def __init__(self, word):
        self.word = word
        self.children = {}

    def add(self, node, distance_func):
        distance = distance_func(self.word, node.word)
        if distance in self.children:
            self.children[distance].add(node, distance_func)
        else:
            self.children[distance] = node

    def search(self, word, max_distance, distance_func):
        distance = distance_func(self.word, word)
        results = []
        if distance <= max_distance:
            results.append(self.word)
        for d in range(max(0, distance - max_distance), distance + max_distance + 1):
            if d in self.children:
                results.extend(self.children[d].search(word, max_distance, distance_func))
        return results

def correct_word_bktree(word, bktree):
    results = bktree.search(word, 2)
    return min(results, key=lambda x: levenshtein_distance(word, x)) if results else word

# n-gram modeli
def words(text): return re.findall(r'\w+', text.lower())

# Eğitim verisini yükleme ve WORDS sözlüğünü oluşturma
with open('training_data.txt') as f:
    WORDS = Counter(words(f.read()))

def P(word, N=sum(WORDS.values())): 
    return WORDS[word] / N

def correction(word): 
    return max(candidates(word), key=P)

def candidates(word): 
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    return set(w for w in words if w in WORDS)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

def correct_word_ngram(word):
    return correction(word)

# Eğitim verilerini oluşturma
def generate_features(word):
    num_vowels = sum(1 for char in word if char in "aeiou")
    num_consonants = len(word) - num_vowels
    return [len(word), num_vowels, num_consonants]

X = []
y = []

training_words = [
    "wod", "attmpt", "exmple", "spel", "algoritm", "corrct", "algortm", "exampl", "spelng", "wrd",
    "definately", "recieve", "seperate", "occured", "untill", "wich", "tommorow", "accomodate", "adress", "arguement",
    "beleive", "calender", "concious", "curiousity", "dependant", "embarass", "enviroment", "existance", "fourty", "goverment",
    "grammer", "harrass", "inoculate", "irresistable", "lollypop", "millenium", "neccessary", "occassion", "parliment", "persistant",
    "posession", "priviledge", "pronounciation", "publically", "questionaire", "refered", "rythm", "sieze", "suprise", "tommorrow",
    "tounge", "twelth", "unfortunatly", "wierd", "writting", "acknowlegde", "acommodate", "amatuer", "arguemint", "carrer",
    "committment", "consistant", "controll", "definately", "dissapear", "embarass", "enviroment", "existense", "experiance",
    "familar", "fianlly", "foriegn", "guage", "hieght", "humerous", "independant", "interupt", "judgement", "knowlege",
    "liason", "medeval", "milennium", "neice", "nieghbor", "occurrance", "omision", "parliment", "perseverence", "posession",
    "prefered", "preform", "promiss", "recieve", "refering", "restaraunt", "reveiw", "seige", "sucess", "tommorow", "truely",
    "unpleaseant", "untill", "wether", "writting", "adress", "arguement", "beleive", "calender", "concious", "curiousity",
    "dependant", "embarass", "enviroment", "existance", "fourty", "goverment", "grammer", "harrass", "inoculate", "irresistable",
    "lollypop", "millenium", "neccessary", "occassion", "parliment", "persistant", "posession", "priviledge", "pronounciation",
    "publically", "questionaire", "refered", "rythm", "sieze", "suprise", "tommorrow", "tounge", "twelth", "unfortunatly", "wierd",
    "writting", "acknowlegde", "acommodate", "amatuer", "arguemint", "carrer", "committment", "consistant", "controll", "definately",
    "dissapear", "embarass", "enviroment", "existense", "experiance", "familar", "fianlly", "foriegn", "guage", "hieght",
    "humerous", "independant", "interupt", "judgement", "knowlege", "liason", "medeval", "milennium", "neice", "nieghbor",
    "occurrance", "omision", "parliment", "perseverence", "posession", "prefered", "preform", "promiss", "recieve", "refering",
    "restaraunt", "reveiw", "seige", "sucess", "tommorow", "truely", "unpleaseant", "untill", "wether", "writting", "adress",
    "arguement", "beleive", "calender", "concious", "curiousity", "dependant", "embarass", "enviroment", "existance", "fourty",
    "goverment", "grammer", "harrass", "inoculate", "irresistable", "lollypop", "millenium", "neccessary", "occassion", "parliment",
    "persistant", "posession", "priviledge", "pronounciation", "publically", "questionaire", "refered", "rythm", "sieze", "suprise",
    "tommorrow", "tounge", "twelth", "unfortunatly", "wierd", "writting"
]

true_words = [
    "word", "attempt", "example", "spell", "algorithm", "correct", "algorithm", "example", "spelling", "word",
    "definitely", "receive", "separate", "occurred", "until", "which", "tomorrow", "accommodate", "address", "argument",
    "believe", "calendar", "conscious", "curiosity", "dependent", "embarrass", "environment", "existence", "forty", "government",
    "grammar", "harass", "inoculate", "irresistible", "lollipop", "millennium", "necessary", "occasion", "parliament", "persistent",
    "possession", "privilege", "pronunciation", "publicly", "questionnaire", "referred", "rhythm", "seize", "surprise", "tomorrow",
    "tongue", "twelfth", "unfortunately", "weird", "writing", "acknowledge", "accommodate", "amateur", "argument", "career",
    "commitment", "consistent", "control", "definitely", "disappear", "embarrass", "environment", "existence", "experience",
    "familiar", "finally", "foreign", "gauge", "height", "humorous", "independent", "interrupt", "judgment", "knowledge",
    "liaison", "medieval", "millennium", "niece", "neighbor", "occurrence", "omission", "parliament", "perseverance", "possession",
    "preferred", "perform", "promise", "receive", "referring", "restaurant", "review", "siege", "success", "tomorrow", "truly",
    "unpleasant", "until", "whether", "writing", "address", "argument", "believe", "calendar", "conscious", "curiosity",
    "dependent", "embarrass", "environment", "existence", "forty", "government", "grammar", "harass", "inoculate", "irresistible",
    "lollipop", "millennium", "necessary", "occasion", "parliament", "persistent", "possession", "privilege", "pronunciation",
    "publicly", "questionnaire", "referred", "rhythm", "seize", "surprise", "tomorrow", "tongue", "twelfth", "unfortunately", "weird",
    "writing", "acknowledge", "accommodate", "amateur", "argument", "career", "commitment", "consistent", "control", "definitely",
    "disappear", "embarrass", "environment", "existence", "experience", "familiar", "finally", "foreign", "gauge", "height",
    "humorous", "independent", "interrupt", "judgment", "knowledge", "liaison", "medieval", "millennium", "niece", "neighbor",
    "occurrence", "omission", "parliament", "perseverance", "possession", "preferred", "perform", "promise", "receive", "referring",
    "restaurant", "review", "siege", "success", "tomorrow", "truly", "unpleasant", "until", "whether", "writing", "address",
    "argument", "believe", "calendar", "conscious", "curiosity", "dependent", "embarrass", "environment", "existence", "forty",
    "government", "grammar", "harass", "inoculate", "irresistible", "lollipop", "millennium", "necessary", "occasion", "parliament",
    "persistent", "possession", "privilege", "pronunciation", "publicly", "questionnaire", "referred", "rhythm", "seize", "surprise",
    "tomorrow", "tongue", "twelfth", "unfortunately", "weird", "writing"
]

# Algoritma isimleri
algorithm_names = ["levenshtein", "bktree", "ngram","damerau_levenshtein","metahpone"]

# BK-Tree'yi oluşturma ve sözlük kelimelerini ekleme
bktree = BKTree(levenshtein_distance)
for word in WORDS:
    bktree.add(word)

for training_word, true_word in zip(training_words, true_words):
    features = generate_features(training_word)
    X.append(features)
    
    # Levenshtein
    start = time.perf_counter()
    correct_word_levenshtein(training_word, true_words)
    time_lev = time.perf_counter() - start
    
    # BK-Tree
    start = time.perf_counter()
    correct_word_bktree(training_word, bktree)
    time_bk = time.perf_counter() - start
    
    # n-gram
    start = time.perf_counter()
    correct_word_ngram(training_word)
    time_ngram = time.perf_counter() - start
    
    # Damerau Levenshtein
    start = time.perf_counter()
    correct_word_damerau_levenshtein(training_word,true_words, 2)
    time_dl = time.perf_counter() - start

    # Metahpone
    start_time = time.perf_counter()
    correct_word_metaphone(training_word,true_words)
    time_mp = time.perf_counter() - start_time

    print(time_lev, time_bk, time_ngram, time_dl,time_mp)

    # En hızlı algoritmayı seçme
    times = [time_lev, time_bk, time_ngram, time_dl,time_mp]
    best_algorithm = algorithm_names[np.argmin(times)]  # Algoritma isimlerini kullanarak etiketleme
    y.append(best_algorithm)

X = np.array(X)
y = np.array(y)

# Decision Tree modelini oluşturma ve eğitme
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Model ve verileri kaydetme
joblib.dump(clf, 'model.joblib')
joblib.dump(bktree, 'bktree.joblib')
joblib.dump(WORDS, 'words.joblib')

# Yeni kelime için en uygun algoritmayı bulma fonksiyonu
def predict_best_algorithm(word):
    features = np.array([generate_features(word)])
    algorithm = clf.predict(features)
    return algorithm[0]