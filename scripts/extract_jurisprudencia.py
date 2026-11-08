import os
import requests
import json
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

AUTHORIZATION = f"Bearer {os.environ['JURIS_API_BEARER_TOKEN']}"
POST_URL = "https://datajuris-sa-dev.lis.ipn.pt/api//importJurisprudencia/jsonlist"
GET_URL_TEMPLATE = "https://datajuris-sa-dev.lis.ipn.pt/api/importJurisprudencia/{}"

FONTES = [
    "TRL",
    "TRC",
    "TRG",
    "TRE",
]

DATA_MINIMA = "2000-01-01"
TAKE = 1000
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "1_raw_plus"
OUT_DIR.mkdir(parents=True, exist_ok=True)

headers = {
    "Authorization": AUTHORIZATION,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

def status(fonte, count):
    sys.stdout.write(f"\r[{fonte}] {count} documentos descarregados...    ")
    sys.stdout.flush()

def checkpoint_path(fonte):
    return OUT_DIR / f"{fonte}.checkpoint"

def load_checkpoint(fonte):
    cp = checkpoint_path(fonte)
    if cp.exists():
        return json.loads(cp.read_text())
    return None

def save_checkpoint(fonte, skip, count, file_pos):
    checkpoint_path(fonte).write_text(json.dumps({"skip": skip, "count": count, "file_pos": file_pos}))

def clear_checkpoint(fonte):
    checkpoint_path(fonte).unlink(missing_ok=True)

for fonte in FONTES:
    out_path = OUT_DIR / f"{fonte}.json"
    cp = load_checkpoint(fonte)

    if cp and out_path.exists():
        skip = cp["skip"]
        count = cp["count"]
        file_pos = cp["file_pos"]
        first = False
        f = open(out_path, "r+b")
        f.seek(file_pos)
        f.truncate()
        sys.stdout.write(f"[{fonte}] Checkpoint encontrado — retomando: skip={skip}, {count} docs já descarregados\n")
        sys.stdout.flush()
    else:
        skip = 0
        count = 0
        first = True
        f = open(out_path, "wb")
        f.write(b"[\n")

    try:
        while True:
            body = {
                "DataMinima": DATA_MINIMA,
                "Fonte": fonte,
                "IdInicio": 0,
                "Skip": skip,
                "Take": TAKE,
            }

            response = requests.post(POST_URL, headers=headers, json=body)
            response.raise_for_status()

            ids = response.json()

            if not ids:
                break

            for item in ids:
                if isinstance(item, dict):
                    id_doc = item.get("Id") or item.get("id")
                else:
                    id_doc = item

                if not id_doc:
                    continue

                for attempt in range(5):
                    try:
                        get_response = requests.get(GET_URL_TEMPLATE.format(id_doc), headers=headers, timeout=30)
                        get_response.raise_for_status()
                        break
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        if attempt == 4:
                            raise
                        wait = 2 ** attempt
                        sys.stdout.write(f"\n[{fonte}] erro de ligação, a aguardar {wait}s...\n")
                        sys.stdout.flush()
                        time.sleep(wait)

                if not first:
                    f.write(b",\n")
                f.write(json.dumps(get_response.json(), ensure_ascii=False, indent=2).encode("utf-8"))
                f.flush()
                first = False
                count += 1
                status(fonte, count)
                time.sleep(0.5)

            skip += TAKE
            save_checkpoint(fonte, skip, count, f.tell())

        f.write(b"\n]")
    finally:
        f.close()

    clear_checkpoint(fonte)
    sys.stdout.write(f"\r[{fonte}] {count} documentos — concluído. Guardado em {out_path}\n")
    sys.stdout.flush()
