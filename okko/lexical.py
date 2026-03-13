import re
import unicodedata

WORD = re.compile(r"[^\W_]+")
STOPWORDS = set(
    "a an and are did does for how in is it of on the to was were what when where which who "
    "а в во где и из как какой кто на о он она по у что это".split()
)


def words(text):
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold().replace("ё", "е"))


def build_lexical(db):
    db.execute("CREATE VIRTUAL TABLE word_search USING fts5(text)")
    db.execute("INSERT INTO word_search(rowid, text) SELECT id, text FROM scenes")


def lexical_hits(db, question):
    tokens = words(question)
    terms = [word for word in tokens if word not in STOPWORDS] or tokens
    terms = list(dict.fromkeys(terms))[:24]
    if not terms:
        return []
    expression = " OR ".join(f'"{term}"' for term in terms)
    return [
        rowid
        for (rowid,) in db.execute(
            "SELECT rowid FROM word_search WHERE word_search MATCH ? ORDER BY rank LIMIT 30",
            (expression,),
        )
    ]
