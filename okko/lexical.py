import re
import unicodedata

WORD = re.compile(r"[^\W_]+")
STOPWORDS = set(
    "a an and are did does for how in is it of on the to was were what when where which who "
    "а в во где и из как какой кто на о он она по у что это".split()
)


def words(text):
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е"))


def trigrams(tokens):
    return (
        token[index:index + 3]
        for token in tokens
        for index in range(len(token) - 2)
    )


def build_lexical(db):
    db.execute("CREATE VIRTUAL TABLE word_search USING fts5(text)")
    db.execute("CREATE VIRTUAL TABLE trigram_search USING fts5(text)")
    for table, tokenize in (("word_search", words), ("trigram_search", lambda text: trigrams(words(text)))):
        rows = db.execute("SELECT id, text FROM scenes")
        db.executemany(
            f"INSERT INTO {table}(rowid, text) VALUES (?, ?)",
            ((rowid, " ".join(tokenize(text))) for rowid, text in rows),
        )


def hits(db, table, terms, limit):
    terms = list(dict.fromkeys(terms))[:limit]
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    return [
        rowid
        for (rowid,) in db.execute(
            f"SELECT rowid FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT 30",
            (expression,),
        )
    ]


def lexical_hits(db, question):
    tokens = words(question)
    terms = [word for word in tokens if word not in STOPWORDS] or tokens
    return (
        hits(db, "word_search", terms, 24),
        hits(db, "trigram_search", trigrams(terms), 80),
    )
