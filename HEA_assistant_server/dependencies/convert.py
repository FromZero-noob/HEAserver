import json
import re
import warnings

def match_json_data(string):
    if "```json" in string:
        json_pattern = r"```json(.*?)```"
        match = re.search(json_pattern, string, re.DOTALL)
        if match:
            code = match.group(1)
        else:
            raise ValueError("No JSON block found in the string.")
        string = re.sub(json_pattern, r"\1", code).replace("\n", "")
    else:
        string = string.replace("\n", "")
        string = string.replace("'", '"')
    return string


def custom_object_hook(dct):

    dct2 = {k: v for k, v in dct.items() if "text" not in k}
    for k, v in dct.items():
        if k not in dct2:
            if isinstance(v, str) and ":" in v:
                v = v.replace(":", "：")
                dct2[k] = v
            else:
                dct2[k] = v
        else:
            dct2[k] = v
    return dct2



def str_to_data(string, del_sign=";@~"):
    if isinstance(string, str):
        if "```json" in string:
            json_pattern = r"```json(.*?)```"
            match = re.search(json_pattern, string, re.DOTALL)
            if match:
                code = match.group(1)
            else:
                raise ValueError("No JSON block found in the string.")
            string = re.sub(json_pattern, r"\1", code).replace("\n", "")
        else:
            string = string.replace("\n", "")
           
        # string = string.replace("'", '’')  

        count = string.count(":")
        if count == 0:
            if ";" in string and ";" in del_sign:
                sts = string.split(";")
                res = [str_to_data(i,del_sign=del_sign) for i in sts]
                return res
            if "@" in string  and "@" in del_sign:
                return tuple(string.split("@"))
            elif "~" in string  and "~" in del_sign:
                return list(string.split("~", maxsplit=1))
            else:
                return string
        else:
            try:
                data = json.loads(string, object_hook=custom_object_hook)
                return str_to_data(data,del_sign=del_sign)
            except json.JSONDecodeError as e:
                try:
                    data = eval(string)
                    return str_to_data(data,del_sign=del_sign)
                except BaseException as e:
                    "{A:B}"  # 仅适合单层字典，单个键值对
                    count = string.count(":")
                    if count == 1:
                        string = string.replace("{", "").replace("}", "").replace('"', "").replace("'", "")
                        string = string.strip()
                        kv = string.split(":", maxsplit=1)
                        if len(kv) == 2:  # dict
                            data = {kv[0]: kv[1]}
                            return str_to_data(data,del_sign=del_sign)
                        elif len(kv) == 1:  # str
                            data = kv[0]
                            return data
                        
                        else:  # error
                            warnings.warn(f"The '{string}', type: {type(string)} is can't be decode.")
                            data = string
                            return data.replace(":", "/")
                    elif isinstance(string, bool):
                        return string
                    else:
                        warnings.warn(f"The '{string}', type:{type(string)} is can't be decode.")
                        return str(string).replace(":", "/")
    elif isinstance(string, dict):
        return {str(str_to_data(str(k).replace(":","/"),del_sign=del_sign)): str_to_data(v,del_sign=del_sign) for k, v in string.items()}
    elif isinstance(string, (list, tuple)):
        res = tuple([str_to_data(v,del_sign=del_sign) for v in string])
        if len(res) == 1:
            return list(res)[0]
        else:
            return res
    
    elif isinstance(string, bool):
        return string
    else:
        warnings.warn(f"The '{string}', type:{type(string)} is can't be decode.")
        return string.replace(":", "/")  # replace ":" to "/" for str


def data_to_str(v, u='', u2='', dec=True, brackets=True, self_str=", "):
    """with str"""
    if isinstance(v, str):
        if v[0] == '"' and v[-1] == '"':
            return v
        else:
            if u != '':
                res = f'{v} {u}'
            else:
                res = v
    elif isinstance(v, (float, int)):
        if u != '':
            res = f'{v} {u}'
        else:
            res = f'{v}'
    elif isinstance(v, (list, tuple)):
        if len(v) == 0:
            return ''
        elif len(v) == 1:
            return data_to_str(v[0], u=u, u2=u2, dec=False)
        elif isinstance(v, tuple):
            if u != '':
                v0 = f'{v[0]} {u}'
            else:
                v0 = v[0]
            if u2 != '':
                v1 = f'{v[1]} {u2}'
            else:
                v1 = v[1]
            res = f'{v0}@{v1}'
            if len(v) > 2:
                vs = "@".join(v[2:])
                res = res + "@" + vs
        elif isinstance(v, list):  # for general
            if isinstance(v[0], tuple):
                data = [data_to_str(i, u=u, u2=u2, dec=False) for i in v]
                res = ";".join(data)
            elif isinstance(v[0], (float, int)) and len(v) == 2:
                
                res = "~".join([str(i) for i in v])
                if u != '':
                    res = f'{res} {u}'
                
            else:  # v[0] str
                data = [data_to_str(i, u=u, u2=u2, dec=False, brackets=False) for i in v]
                res = self_str.join(data)
                res = f'[{res}]'
                return res
        else:
            raise NotImplementedError
    elif isinstance(v, dict):
        st = []
        for ki, vi in v.items():
            st.append(f'{data_to_str(ki, dec=True)}:{data_to_str(vi, u=u, u2=u2, dec=True)}')
        v = ", ".join(st)
        if brackets:
            v = f'{{{v}}}'
        return v
    else:
        res = str(v)

    if dec:
        return f'"{res}"'
    else:
        return f'{res}'


def block_data_to_str(name, prefix, value, unit, brackets=False,**kwargs):
    name = data_to_str(v=name, dec=False)
    if isinstance(unit, str):
        value_str = data_to_str(value, u=unit, dec=False)
    elif isinstance(unit, (list,tuple)):
        if len(unit) == 0:
            u, u2 = '', ''
        elif len(unit) == 1:
            u, u2 = unit[0], ""
        else:
            u, u2 = unit[0], unit[1]
        name = data_to_str(v=name, dec=False)
        value_str = data_to_str(value, u=u, u2=u2, dec=False)
    else:
        raise NotImplementedError
    if kwargs:
        kw_str = data_to_str(kwargs, dec=False, brackets=True)
        if brackets:
            return f'{{"{name}":"{prefix}{value_str}", "kwargs":{kw_str}}}'
        else:
            return f'"{name}":"{prefix}{value_str}", "kwargs":{kw_str}'
    else:
        if brackets:
            return f'{{"{name}":"{prefix}{value_str}"}}'
        else:
            return f'"{name}":"{prefix}{value_str}"'


def btp_data_to_str(name, block, need={},**kwargs):
    name = data_to_str(v=name, dec=False)
    block_str = ", \n    ".join([block_data_to_str(i._key, i._prefix, i._value, i._unit, brackets=False) for i in block]) # fix mutlblock
    
    if need:
        need_str = data_to_str(need, dec=False, brackets=False)
        block_str = f'{block_str}'+', \n    '+f'{need_str}'
    
    if kwargs:
        kw_str = data_to_str(kwargs, dec=False, brackets=True)
        block_str = f'{block_str}'+', \n    '+ f'"kwargs":{kw_str}'
    return f'{{"{name}":{{{block_str}}}}}'


def sequence_data_to_str(name, block, with_name=False, **kwargs):
    res = []
    for blocki in block:
        name = blocki._key
        name = data_to_str(v=name, dec=False)

        block_str = ", \n    ".join([block_data_to_str(i._key, i._prefix, i._value, i._unit, brackets=False) for i in blocki])
        s = f'"{name}":{{{block_str}}}'
        res.append(s)
    
    res_str = ", \n".join(res)
        
    if kwargs:
        kw_str = data_to_str(kwargs, dec=False, brackets=True)
        res_str = f'{res_str}'+', \n    '+ f'"kwargs":{kw_str}'

    if with_name:
        res = f'{{"{name}":{{{res_str}}}}}'
    else:
        res = f'{{{res_str}}}'
        
    return res