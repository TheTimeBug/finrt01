from fastapi import FastAPI

# from V1.client_portfolio import client_portfolio
from mkt_day_end_data import api_adj_data_all
from projectState import projectState_init, projectState_delete

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.get("/api_adj_data_all")
def api_adj_data_all_get():
    result = api_adj_data_all()
    return result


@app.get("/api_adj_data_new")
def api_adj_data_new():
    return {"message": "Hello World"}


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
