# generate_corpus.py
import random

# Sample topics and sentence fragments
subjects = ["I", "You", "We", "They", "He", "She", "Life", "Time", "People", "The world"]
verbs = ["love", "like", "see", "know", "think about", "enjoy", "want", "need", "hate", "create"]
objects = ["books", "music", "movies", "coding", "adventure", "food", "traveling", "learning", "games", "dreams"]
adjectives = ["beautiful", "fun", "exciting", "amazing", "strange", "difficult", "happy", "sad", "colorful", "interesting"]
places = ["home", "school", "work", "park", "city", "village", "universe", "internet", "mountains", "beach"]

lines = []

# Generate 5000 sentences
for _ in range(5000):
    sentence = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(adjectives)} {random.choice(objects)} at {random.choice(places)}."
    lines.append(sentence)

# Save to file
with open("manual_corpus.txt", "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print("manual_corpus.txt generated with 5000 lines!")
