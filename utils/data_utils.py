import os
import sys
import time
import random
import requests
import urllib.request
import zipfile
from pathlib import Path

CACHE_DIR = Path("datasets")
CACHE_DIR.mkdir(exist_ok=True)

CLASS_DATA_FILE = CACHE_DIR / "classification_data.txt"
GEN_DATA_FILE   = CACHE_DIR / "generation_data.txt"

def download_and_cache(url: str, filepath: Path, unzip: bool=False):
    if filepath.exists():
        print(f"[cache] {filepath} already exists, skipping download.")
        return
    print(f"[download] Fetching {url} ...")
    tmpfile, _ = urllib.request.urlretrieve(url)
    if unzip:
        with zipfile.ZipFile(tmpfile, "r") as z:
            z.extractall(CACHE_DIR)
    else:
        os.replace(tmpfile, filepath)
    print(f"[ok] Saved to {filepath}.")

def build_classification_dataset() -> tuple[list[str], list[str]]:
    """Download or load cached edu_class_data from Wikipedia."""
    if CLASS_DATA_FILE.exists():
        lines = CLASS_DATA_FILE.read_text(encoding="utf-8").splitlines()
    else:
        LABEL_CATS = {
            "Math":    "Category:Mathematics",
            "Science": "Category:Science",
            "History": "Category:History",
            "English": "Category:English_language",
        }
        lines = []
        session = requests.Session()
        API = "https://en.wikipedia.org/w/api.php"
        for label, cat in LABEL_CATS.items():
            cm = session.get(API, params={
                "action":"query","list":"categorymembers",
                "cmtitle":cat,"cmlimit":200,"format":"json"
            }).json()
            pageids = [str(m["pageid"]) for m in cm.get("query", {}).get("categorymembers", [])]
            for i in range(0, len(pageids), 20):
                batch = pageids[i : i + 20]
                ex = session.get(API, params={
                    "action":"query","prop":"extracts","exintro":True,
                    "explaintext":True,"pageids":"|".join(batch),"format":"json"
                }).json()
                for page in ex.get("query", {}).get("pages", {}).values():
                    txt = page.get("extract", "").replace("\n", " ").strip()
                    if len(txt) >= 50:
                        lines.append(f"{label}\t{txt}")
                time.sleep(0.1)

        random.shuffle(lines)
        CLASS_DATA_FILE.write_text("\n".join(lines), encoding="utf-8")
        print(f"[cache] Saved {len(lines)} samples to {CLASS_DATA_FILE!r}.")
    # --- FIXED: unpack into (labels, texts) then return (texts, labels) ---
    pairs = [ln.split("\t", 1) for ln in lines]
    labels, texts = zip(*pairs)  
    return list(texts), list(labels)


def build_generation_corpus() -> str:
    """Download or load cached Gutenberg text."""
    if GEN_DATA_FILE.exists():
        return GEN_DATA_FILE.read_text(encoding="utf-8")
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    raw = urllib.request.urlopen(url).read().decode("utf-8", errors="ignore")
    # strip headers
    start = raw.find("*** START")
    end   = raw.find("*** END")
    corpus = raw
    if start != -1 < end:
        corpus = raw[ raw.find("\n", start)+1 : end ]
    GEN_DATA_FILE.write_text(corpus, encoding="utf-8")
    return corpus
