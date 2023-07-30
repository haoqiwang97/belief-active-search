import sqlite3, config

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse

import logging

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# app.mount("/img_database_2d", StaticFiles(directory="./img_database_2d"), name="img_database_2d")

@app.get('/')
def index(request: Request):
    # return {"Hello": "World"}
    img1 = "https://www.w3schools.com/images/picture.jpg" #"/img_database_2d/13268_3D_165_6M_121311_UprightHH1_trim_clean_snapshot_noborder.png"
    img2 = "https://www.w3schools.com/images/w3schools_green.jpg" #"/img_database_2d/13589_3D_166v2_6M_110211_UprightHH1_trim_clean_snapshot_noborder.png"
    # read image to memory

    return templates.TemplateResponse("index.html", {"request": request, "img1": img1, "img2": img2})

@app.get("/home", response_class=HTMLResponse)
def write_home(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})

        # <h1>This is the home page</h1>
        # <h2>USERNAME = test</h2>

        # <form action="/submitform" method="post">
        #     <input type="text" name="assignment">
        #     <input type="text" name="assignment2">
        #     <input type="submit">
        # </form>
    
@app.post("/submitform")
async def add_patient(number: int = Form(...), race: str = Form(...), ethnicity: str = Form(...), age: int = Form(...)):
    #TODO: write to database, use logging
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()

    cursor.execute(
      """
      INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
      """, (number, age, race, ethnicity)
      )
    
    connection.commit()

    logging.info(f"Insert number={number}, race={race}, ethnicity={ethnicity}, age={age}")
    # print(number, race, ethnicity, age)

    return RedirectResponse(url="/patients", status_code=303)
    
# @app.post("/add_patient")
# def add_patient(number: int = Form(...), age: int = Form(...), race: str = Form(...), ethnicity: str = Form(...)):
#     connection = sqlite3.connect(config.DB_FILE)
#     cursor = connection.cursor()

#     cursor.execute(
#     """
#     INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
#     """, (number, age, race, ethnicity)
#     )
#     connection.commit()

#     return RedirectResponse(url=f"/patients", status_code=303)


@app.get("/patients")
def patients(request: Request):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM patient
                   """)
    
    patients = cursor.fetchall()

    return templates.TemplateResponse("patients.html", {"request": request, "patients": patients})