import numpy as np
import joblib
import re
import sys
import json
from collections import Counter
from metaphone import doublemetaphone
import nltk
import time
from spellchecker import SpellChecker
spell = SpellChecker()

new_word = sys.argv[1]

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
    return spell.correction(word)
    if(best_match == None):
        return "No Solution"
    # return best_match

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
    return spell.correction(word)
    # return suggestions

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
    return spell.correction(word)

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
def correct(word):
    corrected_word = spell.correction(word)
    if word != corrected_word:
        return corrected_word
    else:
        return word

def correct_word_bktree(word, bktree):
    results = bktree.search(word, 2)
    return spell.correction(word)
    # return min(results, key=lambda x: levenshtein_distance(word, x)) if results else word

# n-gram modeli
def words(text): return re.findall(r'\w+', text.lower())

# Modeli yükleme
clf = joblib.load('./python/model.joblib')
bktree = joblib.load('./python/bktree.joblib')
WORDS = joblib.load('./python/words.joblib')

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
    return spell.correction(word)
    # return correction(word)

# Yeni kelime için en uygun algoritmayı bulma fonksiyonu
def predict_best_algorithm(word):
    num_vowels = sum(1 for char in word if char in "aeiou")
    num_consonants = len(word) - num_vowels
    features = np.array([[len(word), num_vowels, num_consonants]])
    algorithm = clf.predict(features)
    return algorithm[0]

best_algorithm = predict_best_algorithm(new_word)
# Algoritma fonksiyonları
if best_algorithm == "levenshtein":
    start = time.perf_counter()
    corrected_word = correct_word_levenshtein(new_word, WORDS.keys())
    time_bk = time.perf_counter() - start
elif best_algorithm == "bktree":
    start = time.perf_counter()
    corrected_word = correct_word_bktree(new_word, bktree)
    time_bk = time.perf_counter() - start
elif best_algorithm == "ngram":
    start = time.perf_counter()
    corrected_word = correct_word_ngram(new_word)
    time_bk = time.perf_counter() - start
elif best_algorithm == "damerau_levenshtein":
    start = time.perf_counter()
    corrected_word = correct_word_damerau_levenshtein(new_word,WORDS.keys(), 2)
    time_bk = time.perf_counter() - start
elif best_algorithm == "metahpone":
    start = time.perf_counter()
    corrected_word = correct_word_metaphone(new_word,WORDS.keys())
    time_bk = time.perf_counter() - start

# print(f"Best algorithm for '{new_word}' is algorithm {best_algorithm}")
# print(f"Corrected word: {corrected_word}")

result = {
    "Best Algorithm": (f"Best algorithm for '{new_word}' is algorithm {best_algorithm}"),
    "Corrections": {"word": (f"Corrected word: {(corrected_word)}")},
    "Time":(time_bk)
}

# JSON formatında çıktı
print(json.dumps(result, indent=4))