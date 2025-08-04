from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pipelines.inference_pipeline import run_inference_pipeline
import numpy as np

app = FastAPI()


class InferenceRequest(BaseModel):
    ppk: str
    sc: str
    start_date: Optional[str] = "2021-01-01"
    end_date: Optional[str] = "2025-01-15"


@app.post("/inference")
def inference(request: InferenceRequest):
    try:
        result = run_inference_pipeline(
            ppk=request.ppk,
            sc=request.sc,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        # # If result is a DataFrame or numpy type, convert to JSON serializable
        # if hasattr(result, "to_dict"):
        #     return {"result": result.to_dict(orient="records")}
        # elif isinstance(result, (np.generic, np.ndarray)):
        #     return {"result": result.item()}
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
