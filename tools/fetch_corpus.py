#!/usr/bin/env python3
"""Fetch an English text corpus for vocabulary statistics.
  - Wikipedia: intro sections of random articles + full text of a list of broad, popular topics
  - Project Gutenberg: a few public-domain books
Writes plain-text files into <outdir>. usage: python3 -I fetch_corpus.py <outdir> [n_random_batches]
       python3 -I fetch_corpus.py <outdir> --langs ar,zh,es,fr,de [n_batches]   random intros from other Wikipedias
"""
import json, os, sys, time, urllib.parse, urllib.request

UA = {"User-Agent": "tvbox-vocab-study/0.1 (research script)"}
API = "https://en.wikipedia.org/w/api.php"

def get(url):
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except Exception as e:
            time.sleep(2 + attempt * 3)
    return b""

def wiki_random_intros(n_batches, out, api=API):
    with open(out, "w") as f:
        for b in range(n_batches):
            q = {"action": "query", "format": "json", "generator": "random", "grnnamespace": 0, "grnlimit": 20,
                 "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20}
            d = json.loads(get(api + "?" + urllib.parse.urlencode(q)) or b"{}")
            for p in d.get("query", {}).get("pages", {}).values():
                t = p.get("extract", "").strip()
                if len(t) > 200:
                    f.write(p["title"] + "\n" + t + "\n\n")
            time.sleep(0.2)

TOPICS = ["Australia", "Canberra", "Earth", "Sun", "Moon", "Water", "Photosynthesis", "World War II", "Ancient Egypt",
          "Roman Empire", "Internet", "Computer", "Python (programming language)", "Artificial intelligence",
          "Albert Einstein", "William Shakespeare", "Football", "Music", "Climate change", "Human body", "Cat", "Dog",
          "Cooking", "Mathematics", "Physics", "Chemistry", "Biology", "History of medicine", "Television",
          "Smartphone", "Coffee", "Bread", "Africa", "Europe", "Asia", "United States", "China", "India", "Egypt",
          "Brazil", "Japan", "Germany", "France", "United Kingdom", "Ocean", "Volcano", "Dinosaur", "Solar System",
          "Democracy", "Economics", "Language", "Religion", "Olympic Games", "Electricity", "Car", "Airplane"]

def wiki_topics(out):
    with open(out, "w") as f:
        for t in TOPICS:
            q = {"action": "query", "format": "json", "titles": t, "prop": "extracts", "explaintext": 1, "redirects": 1}
            d = json.loads(get(API + "?" + urllib.parse.urlencode(q)) or b"{}")
            for p in d.get("query", {}).get("pages", {}).values():
                f.write(p.get("title", "") + "\n" + p.get("extract", "") + "\n\n")
            time.sleep(0.2)

BOOKS = {"alice": 11, "pride": 1342, "sherlock": 1661, "frankenstein": 84}

def gutenberg(outdir):
    for name, n in BOOKS.items():
        t = get(f"https://www.gutenberg.org/cache/epub/{n}/pg{n}.txt").decode("utf-8", "replace")
        a, b = t.find("*** START"), t.find("*** END")
        if a > 0 and b > a:
            t = t[t.find("\n", a) + 1:b]
        with open(os.path.join(outdir, f"book_{name}.txt"), "w") as f:
            f.write(t)

if __name__ == "__main__":
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    if len(sys.argv) > 2 and sys.argv[2] == "--langs":
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 30
        for lang in sys.argv[3].split(","):
            out = os.path.join(outdir, f"wiki_random_{lang}.txt")
            wiki_random_intros(n, out, f"https://{lang}.wikipedia.org/w/api.php")
            print(f"{os.path.getsize(out) / 1e6:6.2f} MB  {out}")
        sys.exit(0)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    wiki_topics(os.path.join(outdir, "wiki_topics.txt"))
    gutenberg(outdir)
    wiki_random_intros(n, os.path.join(outdir, "wiki_random.txt"))
    for fn in sorted(os.listdir(outdir)):
        print(f"{os.path.getsize(os.path.join(outdir, fn)) / 1e6:6.2f} MB  {fn}")
