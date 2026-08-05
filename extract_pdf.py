from langchain_community.document_loaders import PyPDFLoader
import os
import re


class ExtractResumePdf:
    def __init__(self,file_path) -> None:
        self.path = file_path

    def extract_pdf(self,path):
        loader = PyPDFLoader(file_path=path)
        docs = loader.load()
        resume_cont = " ".join(page.page_content for page in docs)
        return resume_cont

    def get_candidate_resume_list(self):
        resume_files = [each_file for each_file in os.listdir(self.path)
                        if re.search(r".*\.pdf",each_file)]
        resume_content_list = []
        for each_resume in resume_files:
            resume_data = {}
            resume_data["PDF_file"] = each_resume
            resume_data["Content"] = self.extract_pdf(path=os.path.join(self.path, each_resume))
            resume_content_list.append(resume_data)
        return resume_content_list

# if __name__ == "__main__":
#     path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resume\\")
#     list_resume = ExtractResumePdf(file_path=path).get_candidate_resume_list()
#     print(list_resume[1])