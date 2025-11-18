from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

# from V1.client_portfolio import client_portfolio
from mkt_day_end_data import (
    get_mkt_day_end_data_from_sqlite,
    post_mkt_day_end_data_to_sqlite,
)
from projectState import projectState_delete, projectState_init

app = FastAPI()


# Pydantic model for the incoming data structure
class MarketDayEndData(BaseModel):
    status: int
    message: str
    data: List[
        dict
    ]  # List of dictionaries, can be customized further if structure is known


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/post-mkt-day-end-data")
def post_mkt_day_end_data(payload: MarketDayEndData):
    try:
        # Access the received data
        # status = payload.status
        # message = payload.message
        data_records = payload.data

        result = post_mkt_day_end_data_to_sqlite(data_records)
        if result["status"] == 200:
            return {
                "status": 200,
                "message": "Market day-end data stored successfully",
                "records_count": len(data_records),
            }
        else:
            return {
                "status": 500,
                "message": "Market day-end data stored failed",
                "records_count": len(data_records),
                "data": result,
            }
    except Exception as e:
        return {"status": 500, "message": f"Error processing data: {str(e)}"}


@app.get("/get-mkt-day-end-data/filter={filter_type}/{filter_value}")
def get_mkt_day_end_data(filter_type: str, filter_value: str):
    result = get_mkt_day_end_data_from_sqlite(filter_type, filter_value)

    reply_data = {
        "status": 200,
        "message": "Market day-end data retrieved successfully",
        "records_count": len(result),
        "data": result,
    }
    return reply_data


# initialize the project state
@app.post("/project_init/{project_name}/{password}")
def project_init(project_name: str, password: str):
    if project_name == "rt01" and password == "123456":
        result = projectState_init()
        if result:
            return {
                "message": "Project initialized successfully",
                "project_name": project_name,
            }
        else:
            return {
                "message": "Project initialization failed",
                "project_name": project_name,
            }
    else:
        return {"message": "Project not found or invalid credentials"}


# delete the project state
@app.post("/project_delete/{project_name}/{password}")
def project_delete(project_name: str, password: str):
    if project_name == "rt01" and password == "123456":
        result = projectState_delete()
        if result:
            return {
                "message": "Project deleted successfully",
                "project_name": project_name,
            }
        else:
            return {
                "message": "Project deletion failed",
                "project_name": project_name,
            }
    else:
        return {"message": "Project not found or invalid credentials"}
