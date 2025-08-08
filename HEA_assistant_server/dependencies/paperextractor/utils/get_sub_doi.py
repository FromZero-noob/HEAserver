import requests

doi = "https://doi.org/10.1002/adfm.202203858"
url = f"https://api.crossref.org/works/{doi}"
response = requests.get(url)
data = response.json()
# 获取参考文献
references = data['message'].get('reference', [])

# for i, ref in enumerate(references):
#     cited_doi = ref.get('DOI', 'DOI not found')
#     print(f"{cited_doi}")


with open("sub_doi2.txt", "w") as f:
    for i, ref in enumerate(references):
        print(ref.keys())
        cited_doi = ref.get('DOI', 'DOI not found')
        print(f"{i+1}: {cited_doi}")
        f.write(f"{cited_doi}\n")