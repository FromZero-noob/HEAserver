
import json
from pathlib import Path
from typing import Union


def doi_to_name(dois: Union[str,list], map_file=".per/doi_name_map.json") -> str:
    if isinstance(dois, str):
        dois = [dois]

    map_file = Path(map_file)

    if not map_file.is_absolute():
        map_file = Path.home() / map_file

    print(f"Using DOI to name map file: {map_file}")

    if map_file.exists():
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                doi_map = json.load(f)
        except json.JSONDecodeError:
            # If the file is empty or corrupted, initialize an empty map
            doi_map = {}

    else:
        doi_map = {}

    new = {}

    res = []
    for doi in dois:
        name = doi_map.get(doi, None)
        if name is None:
            # Convert DOI to a name format
            name = doi.replace("/", "-").replace(".", "_").replace(":", "：").replace(">", "").replace("<", "")
            new[doi] = name
        res.append(name)

    if map_file.exists():
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump({**doi_map, **new}, f, ensure_ascii=False, indent=4)
    else:
        map_file.parent.mkdir(parents=True, exist_ok=True)
        with open(map_file, 'w', encoding='utf-8') as f:
            json.dump(new, f, ensure_ascii=False, indent=4)

    return res if len(res) > 1 else res[0]



def name_to_doi(name: Union[str,list],map_file=".per/doi_name_map.json") -> str:
    map_file = Path(map_file)

    if not map_file.is_absolute():
        map_file = Path.home() / map_file

    if map_file.exists():

        with open(map_file, 'r', encoding='utf-8') as f:
            doi_map = json.load(f)
    else:
        doi_map = {}

    if isinstance(name, str):
        name = [name]

    inverse_map = {v: k for k, v in doi_map.items()}

    res = []
    for n in name:
        doi = inverse_map.get(n, None)
        res.append(doi)

    return res if len(res) > 1 else res[0]


if __name__ =="__main__":
    import pandas as pd

    old = pd.read_excel("/root/paperextractor/paperextractor/utils/merged_all_selected_reduced.xlsx",  index_col=None)
    dois =old["doi"].tolist()
    res2 = name_to_doi(dois
                       )
    old["doi"] = res2
    old.to_excel("/root/paperextractor/paperextractor/utils/merged_all_selected_reduced_doi.xlsx", index=False)



