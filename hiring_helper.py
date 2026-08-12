from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from pydantic import BaseModel, Field
from langchain.tools import tool
from langgraph.prebuilt import ToolNode
from extract_pdf import ExtractResumePdf
from typing import List, TypedDict, Dict
import os
import pandas as pd
from llm_wrapper import LLM_wrapper

# Defining Resume Data Schema
class ResumeSchema(BaseModel):
    name : str = Field(description="Extract name of the candiate usually mentioned in the resume header")
    contact_info : List[str] = Field(description="Extract contact information of candidate such has Gmail, Place, LinkedIn, \
                                     phone number Usually mentioned in header after name")
    skills : List[str] = Field(description="Extract candidate main skill sets")
    education : List[str] = Field(description="Extract candidates educations list")
    work_experience : List[str] = Field(description="Candidates work experience")
    summary : str = Field(description="Candidate profile complete summary")
    certifications : List[str] = Field(description="Candidate acquired list of certifications")

class AnalysisSchema(BaseModel):
    rating : int = Field(description="You are expert recruiter, Give a rating to the candidate based given job description between range 1 to 100")
    analysis_summary : str = Field(description="Give brief summary and highlight major points")
    additional_achievements : str = Field(description="Highlight major certifications, achievements or recognitions that is relevent to given job description")
    ratings_sentiment : str = Field(description="Conclude results in this 4 categories Excelent, Good, Average, Not Fit") 

#Defining Resume Agent State
class ResumeAgentState(TypedDict):
    resume_contents : List[Dict]
    candidate_profile : List[Dict]
    analysis : List[Dict]

class ResumeAgent:
    def __init__(self,model,job_description) -> None:
        self.job_description = job_description
        self.normal_llm = model
        self.model_with_schema = model.with_structured_output(ResumeSchema)
        graph = StateGraph(ResumeAgentState)
        graph.add_node("resume_parser",self.resume_parser_node)
        graph.add_node("resume_analyzer",self.resume_analyzer_node)
        graph.add_edge(START,"resume_parser")
        graph.add_edge("resume_parser","resume_analyzer")
        graph.add_edge("resume_analyzer",END)
        self.graph_node = graph.compile()

    def resume_parser_node(self, state: ResumeAgentState):
        resume_cont = state["resume_contents"]
        candidate_profile_list = []
        for each_resume in resume_cont:
            pdf_file = each_resume['PDF_file']
            pdf_cont = each_resume['Content']
            messages = [
                SystemMessage(content="You are expert in Applicant Tracking System (ATS). Given candidates profile Extract name, \
                            contact information, Skills, Education, Work Experience, Summary."),
                HumanMessage(content="Please take resume content given below \
                    {}".format(pdf_cont))
            ]
            response_dict = dict(self.model_with_schema.invoke(messages))
            response_dict['PDF_file'] = pdf_file
            candidate_profile_list.append(response_dict)
        # print("RESPONSE", candidate_profile_list)
        return {"candidate_profile": candidate_profile_list}

    def resume_analyzer_node(self, state: ResumeAgentState):
        analysis_list = []
        candidate_profile = state["candidate_profile"]
        llm_with_analysis_schema = self.normal_llm.with_structured_output(AnalysisSchema)
        for each_candidate in candidate_profile:
            messages = [
                SystemMessage(content=f"You are expert recruiter, Hiring for technical role for software companies. Given Candidate profile \
                    in structured python dictionary, compare with job description provided here {self.job_description} \
                        Respond with depth Analysis"),
                HumanMessage(content=f"Here is the structured candidate profile in python dictionary data type {each_candidate}")
            ]
            response = llm_with_analysis_schema.invoke(messages)
            each_candidate.update(response)
            analysis_list.append(each_candidate)
        return {"analysis": analysis_list}
        


if __name__ == "__main__":

    # # Resume path
    resume_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Resume")

    # Resume data content list
    resume_cont = ExtractResumePdf(file_path=resume_file_path).get_candidate_resume_list()

    # # Job Description Path
    jd_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"JD")
    # # Initilize LLM
    model = LLM_wrapper(llm="ollama",model="gemma4:e2b").choose_llm()
    # # Call ExtractResumePdf to extract resume data content
    # # Job Description file path
    jd = open(jd_file_path+"\\job_description.txt").read()

    # Initilize agent
    agent_1 = ResumeAgent(model=model,job_description=jd)
    response = agent_1.graph_node.invoke({"resume_contents":resume_cont})
    response_analysis = response['analysis']
    df = pd.DataFrame(response_analysis)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Output")
    df.sort_values(by='rating',ascending=False).to_csv(output_path+"\\output.csv",index=False)