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

img_mount_path = "/img_database_2d"
app.mount(img_mount_path, StaticFiles(directory="./img_database_2d"), name="img_database_2d")
app.mount("/temporary", StaticFiles(directory="./temporary"), name="temporary")

def read_imgdb() -> pd.DataFrame:
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
    return df


def random_select() -> tuple[str, str]:
    df = read_imgdb()

    # random select 2 images
    img1, img2 = df.sample(n=2)['img_name'].tolist()

    img1 = img_mount_path + '/' + img1
    img2 = img_mount_path + '/' + img2

    # also write to trials table?
    return img1, img2


def active_select():
    pass


@app.get('/add-patient')
def add_patient():
    pass

@app.post("/submit-patient")
async def submit_patient(number: int = Form(...), language: str = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()

    cursor.execute(
      """
      INSERT INTO patients (number, language) VALUES (?, ?);
      """, (number, language)
      )
    
    connection.commit()

    logging.info(f"Insert number={number}, language={language}")
    # print(number, race, ethnicity, age)

    return RedirectResponse(url="/patientslist", status_code=303)

@app.get("/patientslist")
def patients(request: Request):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM patients
                   """)
    
    patients = cursor.fetchall()

    return templates.TemplateResponse("patientslist.html", {"request": request, "patients": patients})

@app.post("/submit-patient")
async def submit_patient(number: int = Form(...), language: str = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()

    cursor.execute(
      """
      INSERT INTO patients (number, language) VALUES (?, ?);
      """, (number, language)
      )
    
    connection.commit()

    logging.info(f"Insert number={number}, language={language}")
    # print(number, race, ethnicity, age)

    return RedirectResponse(url="/patientslist", status_code=303)

@app.get("/providerslist")
def patients(request: Request):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM providers
                   """)
    
    providers = cursor.fetchall()

    return templates.TemplateResponse("providerslist.html", {"request": request, "providers": providers})

@app.post("/submit-provider")
async def submit_provider(number: int = Form(...), name: str = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()

    cursor.execute(
      """
      INSERT INTO providers (number, name) VALUES (?, ?);
      """, (number, name)
      )
    
    connection.commit()

    logging.info(f"Insert number={number}, name={name}")
    # print(number, race, ethnicity, age)

    return RedirectResponse(url="/providerslist", status_code=303)

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
    # get experiment id, round_count
    experiment_id = 1
    round = 2

    img1 = img1[len(img_mount_path)+1: ]
    img2 = img2[len(img_mount_path)+1: ]

    # name to img_id
    df = read_imgdb()
    img1_id = df.loc[df['img_name'] == img1, 'img_id'].item()
    img2_id = df.loc[df['img_name'] == img2, 'img_id'].item()
 
    if selected_image == "img1left":
        print("Button clicked: img1left")
        select_id = img1_id
    else:
        print("Button clicked: img2right")
        select_id = img2_id

    # write to database
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()
    
    cursor.execute(
    """
    INSERT INTO trials (experiment_id, round, img1_id, img2_id, select_id, timepoint, meanx, meany, stdx, stdy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (experiment_id, round, img1_id, img2_id, select_id, 
          datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
          0.20, -0.23, 0.19, 0.21)
    )

    # todo: update experiments round count
    connection.commit()

    return RedirectResponse(url="/trial", status_code=303)

@app.get("/", response_class=HTMLResponse)
def write_home(request: Request):
    return templates.TemplateResponse("add_patient.html", {"request": request})

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

