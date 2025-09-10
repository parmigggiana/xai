import os
import json
import re
import csv
from typing import Dict, Any, List


def _parse_alpha(alpha_raw: str) -> float | None:
    """Interpreta stringhe tipo '1', '08', '12', '0.8', '1.2' in float.
    Euristiche (dato il naming osservato):
      - Contiene punto -> float diretto
      - Lunghezza 1 -> intero ("1" -> 1.0)
      - Lunghezza 2:
          * Inizia con '0' -> 0.x  ("08" -> 0.8)
          * Altrimenti inserisce un punto dopo il primo char ("12" -> 1.2)
      - Lunghezza >2 senza punto: inserisce punto dopo il primo char ("125" -> 1.25)
    """
    if not alpha_raw:
        return None
    if "." in alpha_raw:
        try:
            return float(alpha_raw)
        except ValueError:
            return None
    if len(alpha_raw) == 1:
        return float(alpha_raw)
    if len(alpha_raw) == 2:
        if alpha_raw.startswith("0"):
            return float(f"0.{alpha_raw[1]}")
        return float(f"{alpha_raw[0]}.{alpha_raw[1]}")
    # lunghezze maggiori (es. '125' -> 1.25)
    return float(f"{alpha_raw[0]}.{alpha_raw[1:]}")


def estrai_parametri_da_nomefile(nome_file: str) -> Dict[str, Any]:
    """Estrae i parametri 'part', 'alpha' e 'weight balancing' (wb) dal nome del file.

    Supporta formati osservati:
      - part1_alpha1.json
      - part2_alpha08_wb-0.json (wb-0 / wb-1)
      - part2_alpha12_wb-1.json
      - part1_alpha0.8_wb-true.json (compatibilità)
    """
    parametri: Dict[str, Any] = {"part": None, "alpha": None, "weight_balancing": None}

    if m := re.search(r"part(\d+)", nome_file):
        parametri["part"] = int(m.group(1))

    # Regex più restrittivo: cattura solo pattern numerici con opzionale parte decimale
    if m := re.search(r"alpha(\d+(?:\.\d+)?)", nome_file):
        alpha_token = m.group(1)
        parsed = _parse_alpha(alpha_token)
        parametri["alpha"] = parsed
        # Warning per ambiguità (nessun punto e lunghezza >1)
        # if "." not in alpha_token and len(alpha_token) > 1:
        #     print(
        #         f"  [warn] alpha '{alpha_token}' in '{nome_file}' interpretata come {parsed}. Usa 'alpha{parsed}' col punto per evitare ambiguità."
        #     )
        if "." not in alpha_token and len(alpha_token) > 1:
            print(
                f"  [debug] alpha ambiguo '{alpha_token}' -> interpretato come {parsed} (file: {nome_file})"
            )

    # Accetta wb-true/false o wb-0/1
    if m := re.search(r"wb-([A-Za-z0-9]+)", nome_file):
        raw = m.group(1).lower()
        if raw in ("true", "false"):
            parsed_wb = raw == "true"
        elif raw.isdigit():
            parsed_wb = int(raw) != 0
        else:
            parsed_wb = None
        parametri["weight_balancing"] = parsed_wb

    # Per i file part1 ignoriamo sempre wb (non valido) anche se presente
    if parametri.get("part") == 1:
        parametri["weight_balancing"] = None

    return parametri


def processa_dati_json_wide(percorso_directory: str) -> List[Dict[str, Any]]:
    """Legge tutti i file JSON di metriche e li normalizza in righe:
        part, alpha, weight_balancing, model, split, dice, hausdorff, source_file

    Attuale struttura JSON osservata:
    {
        "<model_name>": {
            "train": {"dice": .., "hausdorff": ..},
            "val": {..},
            "test": {..}
        },
        ...
    }
    """
    tutte_le_righe: List[Dict[str, Any]] = []

    for nome_file in sorted(os.listdir(percorso_directory)):
        if not (nome_file.endswith(".json") and nome_file.startswith("part")):
            # Salta file non coerenti (es. metrics.json, cache, ecc.)
            continue
        percorso_file_json = os.path.join(percorso_directory, nome_file)
        print(f"Elaborazione di: {nome_file}")

        iperparametri = estrai_parametri_da_nomefile(nome_file)

        try:
            with open(percorso_file_json, "r", encoding="utf-8") as f:
                dati = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  -> Errore lettura/parsing '{nome_file}': {e}. Saltato.")
            continue

        if not isinstance(dati, dict):
            print("  -> Formato inatteso (root non dict). Saltato.")
            continue

        # Livello modello
        for model_name, splits_dict in dati.items():
            if not isinstance(splits_dict, dict):
                continue
            model_lower = model_name.lower()
            part_val = iperparametri.get("part")
            # Filtri richiesti:
            # - Non estrarre modelli 'composite' dai file part2
            if part_val == 2 and "composite" in model_lower:
                continue
            # - Non estrarre modelli 'adaptation' dai file part1
            if part_val == 1 and "adaptation" in model_lower:
                continue
            for split_name, metriche in splits_dict.items():
                if not isinstance(metriche, dict):
                    continue
                riga = {
                    **iperparametri,
                    "model": model_name,
                    "split": split_name,
                    "dice": metriche.get("dice"),
                    "hausdorff": metriche.get("hausdorff"),
                }
                tutte_le_righe.append(riga)

    return tutte_le_righe


def deduplica_righe(righe: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rimuove duplicati con stessa combinazione (part, alpha, weight_balancing, model, split).

    Se trova duplicati con metriche diverse stampa un warning e mantiene la prima occorrenza.
    """
    visti: dict[tuple, Dict[str, Any]] = {}
    for r in righe:
        model_name = r.get("model") or ""
        if model_name.endswith("_2d_finetuned"):
            # Deduplicazione ignorando part/alpha/weight_balancing
            key = ("_FINETUNED_GROUP_", model_name, r.get("split"))
        else:
            key = (
                r.get("part"),
                r.get("alpha"),
                r.get("weight_balancing"),
                model_name,
                r.get("split"),
            )
        if key not in visti:
            visti[key] = r
        else:
            prev = visti[key]
            d1, d2 = prev.get("dice"), r.get("dice")
            h1, h2 = prev.get("hausdorff"), r.get("hausdorff")

            # Confronto con tolleranza per floating point
            def _eq(a, b):
                if a is None or b is None:
                    return a is b
                try:
                    return abs(a - b) <= 1e-12
                except TypeError:
                    return a == b

            if not (_eq(d1, d2) and _eq(h1, h2)):
                print(
                    "[warn] Duplicato con metriche diverse per chiave",
                    key,
                    "; prima dice/hausdorff=",
                    (d1, h1),
                    "; nuova dice/hausdorff=",
                    (d2, h2),
                    " -> mantengo la prima",
                )
            # Altrimenti silenziosamente ignora (duplicato identico)
    return list(visti.values())


if __name__ == "__main__":
    directory_corrente = os.getcwd()
    lista_righe = processa_dati_json_wide(directory_corrente)
    prima = len(lista_righe)
    lista_righe = deduplica_righe(lista_righe)
    dopo = len(lista_righe)
    if dopo < prima:
        print(
            f"Deduplicazione: {prima - dopo} righe duplicate rimosse (totale ora {dopo})."
        )

    if not lista_righe:
        print("Nessun file .json valido trovato nella directory.")
        raise SystemExit(0)

    nome_file_output = "results.csv"
    headers = [
        "part",
        "alpha",
        "weight_balancing",
        "model",
        "split",
        "dice",
        "hausdorff",
    ]

    # Ordina per part, alpha, model, split
    def _sort_key(r: Dict[str, Any]):
        return (
            r.get("part") if r.get("part") is not None else 1e9,
            r.get("alpha") if r.get("alpha") is not None else 1e9,
            r.get("model") or "",
            {"train": 0, "val": 1, "test": 2}.get(r.get("split"), 99),
        )

    lista_righe.sort(key=_sort_key)

    print(f"\nScrittura di {len(lista_righe)} righe nel file '{nome_file_output}'...")
    try:
        with open(nome_file_output, "w", newline="", encoding="utf-8") as file_csv:
            writer = csv.DictWriter(file_csv, fieldnames=headers)
            writer.writeheader()
            writer.writerows(lista_righe)
        print("\nOperazione completata. File creato:", nome_file_output)
    except IOError as e:
        print(f"Errore durante la scrittura del file CSV: {e}")
