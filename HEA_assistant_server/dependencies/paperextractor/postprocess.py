import json
import os
import pathlib

import numpy as np
import pandas as pd

from paperextractor.convert import str_to_data
from paperextractor.utils.doi_to_name import name_to_doi


def expand_dict_columns(file,expand_columns=None,out_dir="output",mark="_doi", old_mark="_raw"):
    """
    :param expand_columns: 需要展开的列名列表，默认为None，表示对所有字典列进行展开
    :return: 展开后的DataFrame
    """
    file = pathlib.Path(file)
    df = pd.read_csv(file, index_col=0, encoding='utf-8')

    if expand_columns is not  None:
        for col in expand_columns:
            if col not in df.columns:
                continue

            sub_df = df[[col]]

            sub_df_columns = sub_df.columns
            sub_df_index = sub_df.index
            sub_df_values = sub_df.values

            dict_mark = []
            values_fmt = []

            for i in sub_df_values:
                i = i[0]
                if isinstance(i, dict):
                    dict_mark.append(True)
                    values_fmt.append(i)

                elif isinstance(i, str) and i.startswith('{') and i.endswith('}') and ":" in i:
                    try:
                        vi = str_to_data(i)
                        if isinstance(vi, dict):
                            dict_mark.append(True)
                            values_fmt.append(vi)
                        else:
                            dict_mark.append(False)
                            values_fmt.append(i)

                    except json.JSONDecodeError:
                        dict_mark.append(False)
                        values_fmt.append(i)
                elif isinstance(i, str) and i in ["","[]","{}","()"]:
                    dict_mark.append(True)
                    values_fmt.append(np.nan)
                elif pd.isna(i):
                    dict_mark.append(True)
                    values_fmt.append(np.nan)
                else:
                    dict_mark.append(False)
                    values_fmt.append(i)

            sub_df = pd.DataFrame(np.array(values_fmt), index=sub_df_index, columns=sub_df_columns, dtype= object)


            if dict_mark.count(False) == 0 and  col in expand_columns:
                # If the column contains dictionaries, flatten them
                flattened = pd.json_normalize(sub_df[col])
                flattened.columns = [f"{col}->{key}" for key in flattened.columns]
                flattened.index = sub_df.index
                sub_df = flattened

            # 合并展开后的DataFrame和原DataFrame
            df = pd.concat([df, sub_df], axis=1)
            # 删除原来的字典列
            df.drop(columns=[col], inplace=True)

    df2 = df.copy()


    # 将包含"->"的列名转换为多级表头
    if any("->" in col for col in df.columns):
        columns = pd.MultiIndex.from_tuples(
            [tuple(col.split("->")) if "->" in col else (col,) for col in df.columns]
        )
        df2.columns = columns

    n_header = df2.columns.nlevels

    out_dir = pathlib.Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    file = file

    if old_mark in file.stem:
        file_stem = file.stem.replace(old_mark,mark, )
    else:
        file_stem = file.stem + mark
    file = out_dir / (file_stem + ".csv")

    df.to_csv(file, index=True, encoding='utf-8')

    df2.to_csv(out_dir / (file_stem + f"_mh_{n_header}.csv"), index=True, encoding='utf-8')

    return file


def add_doi_and_id(file, id_file="id.json", out_dir="output", mark="_doi", old_mark="_raw"):

    file = pathlib.Path(file)
    fn = file.stem
    doi =fn.replace("_raw", "").replace("_doi", "")

    if not os.path.exists(id_file):
        raise FileNotFoundError(f"{id_file} does not exist.")

    with open(id_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        id_val = data["id"]

    df = pd.read_csv(file,index_col=0,encoding='utf-8',)
    df['sample'] = df.index

    id = np.arange(id_val+1, id_val + len(df)+1)

    doi = name_to_doi(doi)

    df["doi"] = doi


    # df["id"] = [f"s-{idi}" for idi in id]

    df.index = [f"s-{idi}" for idi in id]

    id_val = int(id[-1])

    with open(id_file, "w", encoding="utf-8") as f:
        json.dump({"id": id_val}, f)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if old_mark in file.stem:
        file_stem = file.stem.replace( old_mark,mark,)
    else:
        file_stem = file.stem + mark
    file = out_dir / (file_stem  + ".csv")

    df.to_csv(file, encoding='utf-8', index=True)
    print(f"DOI and ID added to {file}.")
    return file


def filter_csv(csv_name, out_dir, old_mark="_expand",mark = "_end",filter_ionic=False,filter_lina=True):
    csv_name = pathlib.Path(csv_name)
    file = csv_name

    if old_mark in file.stem:
        file_stem = file.stem.replace( old_mark,mark,)
    else:
        file_stem = file.stem + mark
    file = out_dir / (file_stem  + ".csv")

    data = pd.read_csv(csv_name, encoding='utf-8',index_col=0)
    data = data.map(lambda x: "" if isinstance(x, dict) and not x else x)
    data = data.map(lambda x: "" if isinstance(x, list) and not x else x)
    data = data.map(lambda x: "" if isinstance(x, tuple) and not x else x)
    data = data.replace(to_replace="[]", value="")
    data = data.replace(to_replace="\{\}", value="")
    data = data.replace(to_replace="{}", value="")
    data = data.replace(to_replace="()", value="")

    if filter_ionic:
        if "ionic conductivity->value" in data.columns:
            filtered_data = data[data["ionic conductivity->value"].notna() & (data["ionic conductivity->value"] != "")]

            filtered_index = [True if "m" in str(i) and "S" in str(i) else False for i in filtered_data["ionic conductivity->value"]]

            filtered_data = filtered_data[filtered_index]

            data = filtered_data

            if len(data) == 0:
                print(f"No valid data found in {csv_name}.")
                return None

    if filter_lina:
        if "composition" in data.columns:
            lina = [True if "Li" in str(i) or "Na" in str(i) else False  for i in data["composition"].values]
            data["Li/Na"] = lina
            print(f"Li/Na column added to {file}.")

    data.to_csv(file, index=True, encoding='utf-8')

    print(f"Filtered data saved to {file}.")
    return file



if __name__ == "__main__":
    # 读取csv文件

    input_csv = "/root/paperextractor/llmtest/res_csv"
    input_csv = pathlib.Path(input_csv)

    for n,i in enumerate(input_csv.iterdir()):


        if "_detail" in i.name:
            continue

        print(i)
        try:

            file_name = add_doi_and_id(i,id_file="/root/paperextractor/id.json",
                                        out_dir="with_doi")
            new_df = expand_dict_columns(file_name, expand_columns=["lattice parameters",
                                                                "ionic conductivity","thermal stability",
                                                                "electronic conductivity","electrochemical window",
                                                                "critical current density",
                                                    "activation energy", "H2O stability",
                                                    "O2 stability","Youngs modulus",
                                                    "shear modulus","volume modulus", "yield strength"],
                                     out_dir="expand_dict")
        except Exception as e:

            pass



