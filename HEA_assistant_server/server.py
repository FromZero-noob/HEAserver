from mcp.server.fastmcp import FastMCP
from dependencies.HEA_extractor import chat_chain_get_msg
from dependencies.HEA_calculator import HEA_calculate
import os
import argparse
from typing import List    # Uncomment if needed
import arxiv
import json
import pathlib
import requests
import re
import joblib
from dotenv import load_dotenv
load_dotenv()
def parse_args():
    """Parse command line arguments for MCP server."""
    parser = argparse.ArgumentParser(description="DPA Calculator MCP Server")
    parser.add_argument('--port', type=int, default=50001, help='Server port (default: 50001)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    try:
        args = parser.parse_args()
    except SystemExit:
        class Args:
            port = 50001
            host = '0.0.0.0'
            log_level = 'INFO'
        args = Args()
    return args
args = parse_args()
mcp = FastMCP("HEA_assistant_server",host=args.host,port=args.port)

#env
HMIX_DATA_PATH =       os.getenv("HMIX_DATA_PATH", "data/binary_Hmix.csv")
SCALER_TYPE_PATH =     os.getenv("SCALER_TYPE_PATH", "models/feats_scaler_type.pkl")
SCALER_STRUC_PATH =    os.getenv("SCALER_STRUC_PATH","models/feats_scaler_struc.pkl")
SCALER_Tm_PATH =       os.getenv("SCALER_Tm_PATH","models/feats_scaler_Tm.pkl")
TYPE_MODEL_PATH =      os.getenv("TYPE_MODEL_PATH", "models/Type_predict_model_XGB.pkl")
STRUCTURE_MODEL_PATH = os.getenv("STRUCTURE_MODEL_PATH", "models/Structure_predict_model.pkl")
Tm_MODEL_PATH =        os.getenv("Tm_MODEL_PATH","models/Tm_predict_model_xgb.pkl")

#load models
tm_scaler = joblib.load(SCALER_Tm_PATH)
tm_model = joblib.load(Tm_MODEL_PATH)
type_scaler = joblib.load(SCALER_TYPE_PATH)
struc_scaler = joblib.load(SCALER_STRUC_PATH)
type_model = joblib.load(TYPE_MODEL_PATH)
struc_model = joblib.load(STRUCTURE_MODEL_PATH)
PAPER_DIR = 'search_paper'
@mcp.tool()
def search_paper(user_query: str,search_type:str, max_results: int = 3) -> dict:
    """
    Search for papers on arXiv by title, author or keywords,
    download the original publications and save basic information.
    read the info file to give information detials
    Args:
        user_query: string
        query_type: 'title', 'author', or 'all'
        max_results: Maximum number of results to retrieve (default: 5)
    Returns:
        path: The directory where papers and info are saved.
    """
    # Use arxiv to find the papers 
    client = arxiv.Client()
    # Search for papers where the title/author/all matches the input
    if  search_type == 'title':
        search_query = f'ti:"{user_query}"'
    elif search_type == 'author':
        search_query = f'au:"{user_query}"'
    elif search_type == 'all':
        search_query = f'all:"{user_query}"'
    else:
        raise ValueError("search_type must be 'title', 'author', or 'all'")
    
    search = arxiv.Search(
               query=search_query,  
               max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )  
    papers = client.results(search)

    # Directory management (optional, like original function)
    path = os.path.join(PAPER_DIR, user_query.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    info_path = os.path.join(path, "papers_info.json")
    # Try to load existing papers info
    try:
        with open(info_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}
    # Process and download each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        safe_title = paper.title[:50]
        pdf_filename = re.sub(r'[-\\/*?:"<>| ]', "_", safe_title) + ".pdf"
        file_path = os.path.join(path,pdf_filename)
        print(f"Downloading {paper.pdf_url} -> {file_path}")
        try:
            pdf_resp = requests.get(paper.pdf_url)
            if pdf_resp.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(pdf_resp.content)
                print(f"Saved: {file_path}")
                paper_info['pdf_path']=file_path
            else:
                print(f"Failed to download {paper.pdf_url}: {pdf_resp.status_code}")
                paper_info['pdf_path']=None
        except Exception as e:
                print(f"Error downloading {paper.pdf_url}: {e}")
                paper_info['pdf_path']=None
        papers_info[paper_id] = paper_info

    # Save updated papers_info to json file
    with open(info_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2, ensure_ascii=False)
    print(f"Results and PDFs are saved in: {path}")
    return papers_info
@mcp.tool()
def HEA_params_calculator(name:str)->dict:
    '''Split chemical formula, and calculate VEC, delta, Hmix, Smix, Lambda parameters'''
    df = HEA_calculate(name,HMIX_DATA_PATH)
    return df.to_dict()
@mcp.tool()
def HEA_data_extract(manuscript_path:str, out_dir:str)->str:
    '''
    Args: 
       manucript: .pdf file path of the manuscript to be analysed
       out_dir: output path
    
    extract structural data of High Entropy Alloy from a manuscript
    including: name,composition,detailed phase structure,mechanical properties
    save results as csv file
    return the csv file path'''
    if os.path.isfile(manuscript_path) and manuscript_path.lower().endswith('.pdf'):
        # 单个PDF文件，直接处理
        return chat_chain_get_msg(
            manuscript=manuscript_path,
            support_si_dir=None,  # 如有补充材料可指定目录
            out_dir=out_dir,
            del_old=True,
            debug=False,
            model_json=None,      # 若有已生成的json，可加速流程
            verbose=True,
            warm_start=False,
            warm_prompt=None
        )
    elif os.path.isdir(manuscript_path):
        # 文件夹，批量处理所有pdf文件
        pdf_files = glob.glob(os.path.join(manuscript_path, "*.pdf"))
        if not pdf_files:
            return f"No PDF files found in directory: {manuscript_path}"
        results = {}
        for pdf in pdf_files:
            try:
                result = chat_chain_get_msg(
                    manuscript=pdf,
                    support_si_dir=None,
                    out_dir=out_dir,
                    del_old=True,
                    debug=False,
                    model_json=None,
                    verbose=True,
                    warm_start=False,
                    warm_prompt=None
                )
                results[os.path.basename(pdf)] = result
            except Exception as e:
                results[os.path.basename(pdf)] = f"Error: {e}"
        return json.dumps(results, indent=2, ensure_ascii=False)
    else:
        return f"Input {manuscript_path} is neither a PDF file nor a directory."
@mcp.tool()
def HEA_predictor(name:List[str])->List[str]:
    '''1.Split the chemical formula of a High Entropy Alloy, receive List of strings and convert each string to a dataframe.
         Columns are the name of elements included,adn values are the corresponding molar ratio, and then filter the elements not included
       2.Calculated certain parameters and add to the dataframe
       3.Calculated if certain rules are satisfied, add the results to the dataframe
       4.Using the dataframe and a RF model (trained on a provided dataset) to predict if the formula can form a solid-solution system, if so, predict its crystal structure'''
    #features_Tm = df.iloc[:,1:]
    df = HEA_calculate(name,HMIX_DATA_PATH)
    X_scaled = tm_scaler.transform(df)
    Tm_pred =  tm_model.predict(X_scaled)
    df['Tm'] = Tm_pred
    for index, row in df.iterrows():
        df.at[index, "Omega"] = row['Tm']*row['Smix(J/Kmol)']/(abs(row['Hmix(kJ/mol)'])*1000)
    #load model and predict------------------------------------------------------------------------------------------------
    X_pred = df.iloc[:,:42]
    X_pred_scaled = type_scaler.transform(X_pred)
    res = []
    y_pred_type = type_model.predict(X_pred_scaled)
    if y_pred_type[0] == 1:
        res.append ('solid solution')
    elif y_pred_type[0] == 2:
        res.append ('Solid Solution + Intermetallic compounds')
    elif y_pred_type[0] == 0:
        res.append ('Intermetallic compounds')

    if (res[0] == 'solid solution'): 
       X_pred_scaled = struc_scaler.transform(X_pred)
       y_pred_2 = struc_model.predict(X_pred_scaled)
       if y_pred_2[0] == 0:
          res.append('BCC')
       elif y_pred_2[0] == 2:
          res.append('FCC')
       elif y_pred_2[0] == 1:
          res.append('FCC+BCC')
       else:
          res.append('complicated structure')
    return(res)

if __name__ == "__main__":
    mcp.run(transport="sse")
