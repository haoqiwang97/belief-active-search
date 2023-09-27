# uvicorn main:app --reload 
import sqlite3, config

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse

import logging
from datetime import datetime

import pandas as pd


app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/img_database_2d", StaticFiles(directory="./img_database_2d"), name="img_database_2d")
app.mount("/temporary", StaticFiles(directory="./temporary"), name="temporary")

def random_select() -> tuple[str, str]:
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM imgdb
                   """)
    
    imgdb = cursor.fetchall()

    column_names = [description[0] for description in cursor.description]
    df = pd.DataFrame(imgdb, columns=column_names)

    # random select 2 images
    img1, img2 = df.sample(n=2)['img_name'].tolist()

    img1 = '/img_database_2d/' + img1
    img2 = '/img_database_2d/' + img2

    # also write to trials table?
    return img1, img2


def active_select():
    pass


@app.get('/trial')
def trial(request: Request):
    # read image to memory
    # todo: can do it without mount?
    # select 2 images
    # manually select
    img1 = "/img_database_2d/13268_3D_165_6M_121311_UprightHH1_trim_clean_snapshot_noborder.png" # "https://www.w3schools.com/images/picture.jpg" #
    img2 = "/img_database_2d/13589_3D_166v2_6M_110211_UprightHH1_trim_clean_snapshot_noborder.png" # "https://www.w3schools.com/images/w3schools_green.jpg" #
    
    # random select
    img1, img2 = random_select()
    pred = "/temporary/search.png"
    return templates.TemplateResponse("trial.html", {"request": request, "img1": img1, "img2": img2, "pred": pred})

@app.post("/submit-trial")
async def submit_trial(selected_image: str = Form(...), img1: str = Form(...), img2: str = Form(...)):
    experiment_id = 1
    round = 2
    # todo: read database to know which images are shown?
    # img1 = 2
    # img2 = 3
    print(img1, img2)
    # todo: name to img_id
    if selected_image == "img1left":
        print("Button clicked: img1left")
        selection = img1
    else:
        print("Button clicked: img2right")
        selection = img2

    # write to database
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()
    
    cursor.execute(
    """
    INSERT INTO trials (experiment_id, round, img1, img2, selection, timepoint, meanx, meany, stdx, stdy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (experiment_id, round, img1, img2, selection, 
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
          0.20, -0.23, 0.19, 0.21)
    )

    connection.commit()

    return RedirectResponse(url="/trial", status_code=303)

@app.get("/", response_class=HTMLResponse)
def write_home(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})
    
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