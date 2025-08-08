
import functools
import pathlib
import pandas as pd
from typing import Union
from PyPDF2 import PdfReader
import pathlib
from multiprocessing import Pool


def get_pdf_text(pdf: Union[str, pathlib.Path]):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += "\n"
        text += page.extract_text()

    print(f"Paper ({pdf}) words number :{len(text)}")
    return text


def get_pdf_text_save(pdf: Union[str, pathlib.Path],new_dir):
    new_dir = pathlib.Path(new_dir)
    new_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf = pathlib.Path(pdf)
        text = get_pdf_text(pdf)
        name = pdf.name
        name = name.replace(".pdf", ".txt")
        with open(new_dir/name, "w") as f:
            f.write(text)
        print(pdf)
    except Exception as e:
        print(f"Error processing {pdf}: {e}")
    
    
def process_pdf_batch(pdf_dir, new_dir):


    process_pdf = functools.partial(get_pdf_text_save, new_dir=new_dir)

    pdf_dir = pathlib.Path(pdf_dir)
    pdfs = list(pdf_dir.rglob("*.pdf"))
    pdfs = [str(i) for i in pdfs]

    with Pool(16) as pool:
        pool.map(process_pdf, pdfs)
    pool.close()
    pool.join()

    
def filter_txt_func(txt,filter_txt=["ionic conductivity"],logic="and"):
    
    if isinstance(filter_txt, str):
        filter_txt = [filter_txt]
        
    
    def not_f(x):
        if isinstance(x, bool):
            return not x
        else:
            return not all(x)
      
    def all_f(x):
        if isinstance(x, bool):
            return x
        else:
            return all(x)  
    
    def or_f(x):
        if isinstance(x, bool):
            return x
        else:
            return any(x)
    
    try:
        with open(txt, "r") as f:
            ftxt = f.read()
    except Exception as e:
        raise e
        print(f"Error processing {txt}: {e}")
        return False
    
    marks=[]
    for fii in filter_txt:
        if fii in ftxt:
            mark = True
        else:
            mark = False
        marks.append(mark)
        
    mapss = {"and": all_f, "or": or_f, "not": not_f,"must": all_f}
    
    if logic in mapss:
        cmd = mapss[logic]
        if cmd(marks):
            return True
        else:
            return False
    elif isinstance(logic, (tuple,list)):
        if "and" in logic:
            assert "or" not in logic, "or, and cannot be used at the same time"
        elif "or" in logic:
            assert "and" not in logic, "or, and cannot be used at the same time"
            
        cmdss = []
        for logicii in logic:
            if logicii in mapss:
                cmd = mapss[logicii]
            else:
                raise ValueError(f"Invalid logic: {logic}")
            cmdss.append(cmd)
            
        if "and" in logic:
            if all([cmd(mark) for cmd, mark in zip(cmdss, marks)]):
                return True
            else:
                return False
        elif "or" in logic:
            ms = True
            res = []
            for cmd, mark,lg in zip(cmdss, marks,logic):
                if "must" in lg or "not" in lg:
                    ms = cmd(mark)*ms
                else:
                    res.append(cmd(mark))
            if any(res)*ms:
                return True
            else:
                return False
        else:
            return not_f(marks)
        
    raise ValueError(f"Invalid logic: {logic}")

def filter_txts_batch(txts, filter_txt=["ionic conductivity"], logic="and"):
    
    
    func = functools.partial(filter_txt_func, filter_txt=tuple(filter_txt), logic=tuple(logic) if isinstance(logic, list) else logic)
    
    with Pool(16) as pool:
        res = pool.map(func=func, iterable=txts)
    res = list(res)
    pool.close()
    pool.join()
    
    name = [pathlib.Path(txt).stem for txt in txts]
    
    data = pd.DataFrame({"path": txts,"name":name, "mark": res})
    data.to_csv("filter.csv", index=False)
    data[data["mark"]==True].to_csv("filter_true.csv", index=False)
    
    with open("filter_true.txt", "w") as f:
        for i in data[data["mark"]==True]["path"]:
            f.write(str(i) + "\n")
    print("filter success number: ", len(data[data["mark"]==True]))
    


# if __name__ == "__main__": 

#     import pathlib
#     from multiprocessing import Pool

#     pttxt = pathlib.Path("/dp-library/solid_electrolyte/txt01")
#     txts = list(pttxt.glob("*.txt"))
#     txts = [str(i) for i in txts]
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity"], logic="and")
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","temperature"], logic=["and","and"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","space group"], logic=["and","and"])
#     filter_txts_batch(txts, filter_txt=["ionic conductivity","space group","amorphous"], logic=["must","or","or"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","amorphous"], logic=["and","and"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","cm-1"], logic=["and","not"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","cm−1"], logic=["and","and"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","crystal"], logic=["and","and"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","℃","°C"], logic=["must","or","or"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","℃","°C","temperature"], logic=["must","or","or","or"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","℃","°C","amorphous","temperature"], logic=["must","or","or","must","or"])
#     # filter_txts_batch(txts, filter_txt=["ionic conductivity","℃"], logic=["and","and"])

def add_args(parser):
    parser.add_argument("--pdf_dir", type=str,help="Directory containing PDF files.",default=None)
    parser.add_argument("--txt_dir", type=str, help="Directory containing text files.", default=None)
    parser.add_argument("--txts", type=str, help="file containing text files.", default=None)
    parser.add_argument("--new_dir", type=str,  help="Directory to save extracted text files.")
    parser.add_argument("--filter_txt", type=str, nargs='+', default="ionic conductivity", help="Keywords to filter text files.")
    parser.add_argument("--logic", type=str, default="and",nargs='+',  help="Logic for filtering (and/or/not).")
    
    parser.add_argument("-c", "--config", type=str,  help="Path to configuration json file.")

    return parser

def main(args):
    import argparse
    parser = argparse.ArgumentParser(description="Filter PDF files and extract text.")
    parser = add_args(parser)
    args = parser.parse_args(args)
    
    
def run(args):
    if args.config:
        import json
        with open(args.config, "r") as f:
            config = json.load(f)
        args.__dict__.update(config)
    
    for k,v in args.__dict__.items():
        print(f"Argument {k} = {v}")
        
    if args.txts:
        txts = pathlib.Path(args.txts)
        with open(txts, "r") as f:
            txts = f.readlines()
        txts = [i.strip() for i in txts if i.strip()]
    else:
        assert args.pdf_dir is not None or args.txt_dir is not None, "Please provide at least one of --pdf_dir or --txt_dir."
        if args.pdf_dir:
            process_pdf_batch(args.pdf_dir, args.new_dir)
            txt_dir = pathlib.Path(args.new_dir)
            
        elif args.txt_dir:
            txt_dir = pathlib.Path(args.txt_dir)
            
        txts = list(txt_dir.rglob("*.txt"))
     
    filter_txts_batch(txts=txts, filter_txt=args.filter_txt, logic=args.logic)

if __name__ == "__main__":

    main()
   
    
    