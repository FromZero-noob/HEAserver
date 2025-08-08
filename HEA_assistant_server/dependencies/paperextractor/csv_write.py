import argparse
import functools
import json
import multiprocessing as multip
import pathlib
from copy import deepcopy
import re 

import pandas as pd
from tqdm import tqdm

from llmchat.chater import BaseAssistant
from paperextractor.convert import str_to_data
from paperextractor.postprocess import (
    add_doi_and_id,
    expand_dict_columns,
    filter_csv,
)
def _data_init_to_csv(data):
    data0, data_01, data_02,jdata_03, data04, data_05, data_06, data_07, data_08, data1, data__2, data2, data__3, data3, data__4, data4, data__5, data5, data__6, data6, data__7, data7, *_ = data
    #前几个元素一次赋值，*_捕捉剩余元素并忽略

    print("data0:", data0)
    print("data01:", data_01)
    print("data02:", data_02)
    print("data1:", data1)
    print("data2:", data2)
    print("data3:", data3)
    print("data4:", data4)
    print("data5:", data5)
    print("data6:", data6)
    print("data7:", data7)

    data1 = str_to_data(data1, del_sign="")#str_to_data
    data2 = str_to_data(data2, del_sign="")
    data3 = str_to_data(data3, del_sign="")
    data4 = str_to_data(data4, del_sign="")
        
    data5 = str_to_data(data5)
    data6 = str_to_data(data6, del_sign="")
    data7 = str_to_data(data7, del_sign="")

    # pop 出 data2 的general_1, general_2, ..., 并合并到data2中
    if isinstance(data2, dict):
        general_keys = [k for k in data2.keys() if k.startswith("general_")]

        vs = [data2.pop(k, {}) for k in general_keys]
        for v in vs:
            if isinstance(v, dict):
                match_list = v.pop("match list", [])
                for sub_k in match_list:
                    if sub_k in data2:
                        data_ = {}
                        data_.update(v)
                        data_.update(data2[sub_k])
                        data2[sub_k]= data_
                    else:
                        data2[sub_k] = deepcopy(v)

    if isinstance(data7, dict):
        general_keys = [k for k in data7.keys() if k.startswith("general")]

        vs = [data7.pop(k, {}) for k in general_keys]
        for v in vs:
            if isinstance(v, dict):
                for d7k,d7v in data7.items():
                    if isinstance(d7v, dict):
                        for sub_k, sub_v in d7v.items():
                            if sub_k in v:
                                if isinstance(v[sub_k], dict):
                                    data_ = {}
                                    data_.update(v[sub_k])
                                    data_.update(sub_v)
                                    data7[d7k][sub_k] = data_
                                else:
                                    data7[d7k][sub_k] = deepcopy(v[sub_k])



    samp_tems = {}
    print("1111111111111111")
    print(type(data5))
    for k,v5 in data5['content'].items():
        base_name = v5.get("base name",None)
        source5 = v5.get("source text","")
        if base_name is None:
           continue
        term = {k:{"source text":"","base name":base_name}}
        # print(base_name)
        if base_name in data1:
            v1 = deepcopy(data1[base_name])
            temp_text = v1.pop("source text","")
            term[k].update(v1)
            term[k]["source text"] = "".join((term[k]["source text"], str(temp_text)))
        if base_name in data2:
            v2 = deepcopy(data2[base_name])
            temp_text = v2.pop("source text","")
            term[k].update(v2)
            term[k]["source text"] = "".join((term[k]["source text"], str(temp_text)))
        chain = v5.get("process chain",[])
        process_chain_detail= {}
        for i in chain:
            if i in data3:
                process_chain_detail[i] = deepcopy(data3[i])
        term[k]["process chain detail"] = process_chain_detail
        term[k]["process chain"] = chain
        term[k]["source text"] = "".join((term[k]["source text"], str(source5)))
        term[k]["states"] = v5.get("states","")

        if k in data7:
            v7 = deepcopy(data7[k])
            source7 = v7.pop("source text","")
            term[k].update(v7)
            term[k]["source text"] = "".join((term[k]["source text"],str(source7)))

        # print(term)
        samp_tems.update(term)

    add_samples = {}

    old_name = []

    for ks,vs in samp_tems.items():
        for kp,vp in vs.items():
            if isinstance(vp, dict):
                vpv = [k for k in vp.keys()]
                vpv_index = ["value@" in k  for k in vpv]
                try:
                    if len(vpv_index) > 0 and any(vpv_index):
                        vpvk = vpv[vpv_index.index(True)] # 获取第一个包含"value@"的键
                        vpvv = vp[vpvk] # 获取对应的值

                        tpk = vpvk.replace("value@", "")
                        if "@" in vpvk and ";" in vpvv and "@" in vpvv:

                            if len(tpk) == 1:
                                tpk = str(vpvk).split("@")[1]
                                vpvvs = str(vpvv).split(";")
                                tpv = [vpvvi.split("@") for vpvvi in vpvvs]

                                for j,(tpvi, tpvv) in enumerate(tpv):
                                    n_vs = deepcopy(vs)
                                    n_vs[kp].pop(tpk, None)
                                    n_vs[kp].pop(vpvk, None)
                                    n_vs[kp]["value"] = tpvi
                                    n_vs[kp][tpk] = tpvv
                                    add_samples[f"{ks}_{tpk}_{j}"] = n_vs
                                    old_name.append(ks)
                            else:
                                tpk = str(vpvk).split("@")[:]
                                vpvvs = str(vpvv).split(";")
                                tpv = [vpvvi.split("@") for vpvvi in vpvvs]

                                for j,tpvi in enumerate(tpv):
                                    n_vs = deepcopy(vs)
                                    n_vs[kp].pop(vpvk, None)
                                    for tpkii, tpvii in zip(tpk, tpvi):
                                        n_vs[kp][tpkii] = tpvii
                                    tpkstr = "_".join(tpk)
                                    add_samples[f"{ks}_{tpkstr}_{j}"] = n_vs
                                    old_name.append(ks)

                except Exception as e:
                    pass
    
    if len(samp_tems) == 0:
        print ('222')

    #for i in old_name:
        #if i in samp_tems:
           # samp_tems.pop(i, None)
    
    samp_tems.update(add_samples)

    if len(samp_tems) == 0:
        print ('111')
        return None
    else:
        data = pd.DataFrame.from_dict(samp_tems, orient="index")
        data = data.map(lambda x: "" if isinstance(x, dict) and not x else x)
        data = data.map(lambda x: "" if isinstance(x, list) and not x else x)
        data = data.map(lambda x: "" if isinstance(x, tuple) and not x else x)
        data = data.replace(to_replace="[]", value="")
        data = data.replace(to_replace="\{\}", value="")
        data = data.replace(to_replace="{}", value="")
        data = data.replace(to_replace="()", value="")

        return data




# 1. 读取json数据
with open('D:\\DPTechnology\\paperextractor\\manuscripts\\1-s2.0-S1000681824000043-main\\0_model_json\\raw.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# 2. 假设data_init_to_csv需要的是具体一条数据（如json_data[0]），你可以遍历
df = _data_init_to_csv(json_data['messages'])  # 或传多个，依照你的数据格式

print(df)

# 3. 保存为csv
df.to_csv('D:\\DPTechnology\\paperextractor\\manuscripts\\1-s2.0-S1000681824000043-main\\result.csv', index=False)