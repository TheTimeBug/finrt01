import json
from typing import List

from fastapi import FastAPI, Request
from pydantic import BaseModel

from helper import get_last_entry_from_sqlite
from lvarCal import lvrCal
from mkt_day_end import (
    get_mkt_day_end_data_from_sqlite,
    post_mkt_day_end_data_to_sqlite,
)
from mkt_security import get_mkt_security_from_sqlite, post_mkt_security_to_sqlite
from projectState import projectState_delete, projectState_init

app = FastAPI()


# Pydantic model for the incoming data structure
class projectDataInit(BaseModel):
    status: int
    message: str
    data: List[
        dict
    ]  # List of dictionaries, can be customized further if structure is known


class syncData(BaseModel):
    status: int
    message: str
    mde: List[dict]
    msi: List[dict]
    csh: List[dict]
    # List of dictionaries, can be customized further if structure is known


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/get-last-entry")
def get_last_entry():
    result = get_last_entry_from_sqlite()
    return result


@app.post("/get-lvar-data")
async def get_lvar_data(request: Request):
    # Handle both JSON object and JSON string
    try:
        body = await request.body()
        body_str = body.decode("utf-8")

        # Try to parse as JSON
        try:
            data = json.loads(body_str)
        except json.JSONDecodeError:
            # If it fails, it might already be a dict
            data = body_str

        # If data is still a string (double-encoded), parse again
        if isinstance(data, str):
            data = json.loads(data)

        # Validate with Pydantic
        payload = syncData(**data)

    except Exception as e:
        return {
            "status": 400,
            "message": f"Invalid request format: {str(e)}",
        }

    result_mde = post_mkt_day_end_data_to_sqlite(payload.mde)
    result_msi = post_mkt_security_to_sqlite(payload.msi)

    if result_mde["status"] == 200 and result_msi["status"] == 200:
        result_lvrCal = lvrCal(payload.csh)
        if result_lvrCal["status"] == 200:
            return {
                "status": 200,
                "message": "Data stored and LVR calculated successfully",
                "data": result_lvrCal,
            }
        else:
            return {
                "status": 500,
                "message": "LVR calculation failed",
            }
    else:
        return {
            "status": 500,
            "message": "Data stored failed",
            "mde": result_mde,
            "msi": result_msi,
        }


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


@app.get("/get-mkt-security-data")
def get_mkt_security_data():
    result = get_mkt_security_from_sqlite()
    reply_data = {
        "status": 200,
        "message": "Market security data retrieved successfully",
        "records_count": len(result),
        "data": result,
    }
    return reply_data


# project state related functions
# for initialize,delete and sync mde and msi the project state
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


@app.post("/project_data_init_mde")
def project_data_init(payload: projectDataInit):
    result = post_mkt_day_end_data_to_sqlite(payload.data)
    if result["status"] == 200:
        return {
            "message": "Project data initialized successfully",
            "data_count": result["data_count"],
        }
    else:
        return {
            "message": "Project data initialization failed",
            "data_count": result,
        }
