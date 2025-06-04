from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from doc_draft import (
    WritPetitionRequest, AffidavitRequest, PatentApplicationRequest,
    AnnexureRequest, WitnessStatementRequest, ExhibitRequest,
    ForensicReportRequest, ExpertOpinionRequest,
    generate_writ_petition, generate_affidavit, generate_patent_application,
    generate_annexure, generate_witness_statement, generate_exhibit,
    generate_forensic_report, generate_expert_opinion
)

app = FastAPI(
    title="AI Paralegal API",
    description="API for generating various legal documents using AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to AI Paralegal API"}

# Include all endpoints from doc_draft.py
app.post("/generate/writ_petition")(generate_writ_petition)
app.post("/generate/affidavit")(generate_affidavit)
app.post("/generate/patent_application")(generate_patent_application)
app.post("/generate/annexure")(generate_annexure)
app.post("/generate/witness_statement")(generate_witness_statement)
app.post("/generate/exhibit")(generate_exhibit)
app.post("/generate/forensic_report")(generate_forensic_report)
app.post("/generate/expert_opinion")(generate_expert_opinion)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)