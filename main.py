# uvicorn main:app --reload 
import sqlite3, config

from fastapi import FastAPI, Request, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse

import logging
from datetime import datetime
from datetime import date  # Import the date type

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

@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

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

def read_patients():
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM patients
                   """)
    
    patients = cursor.fetchall()
    return patients

def read_providers():
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM providers
                   """)
    
    providers = cursor.fetchall()
    return providers

def read_participants():
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM participants
                   """)
    
    participants = cursor.fetchall()
    return participants

@app.get("/participantslist")
def participants(request: Request):    
    patients = read_patients()
    providers = read_providers()
    participants = read_participants()
    return templates.TemplateResponse("participantslist.html", {"request": request, "patients": patients, "providers": providers, "participants": participants})

@app.post("/submit-participant")
async def submit_participant(selected_patient: int = Form(...), selected_provider: int = Form(...), type: str = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()
    cursor.execute(
    """
    INSERT INTO participants (type, patient_id, provider_id) VALUES (?, ?, ?);
    """, (type, selected_patient, selected_provider)
    )
    connection.commit()

    return RedirectResponse(url="/participantslist", status_code=303)

def read_visits():
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM visits
                   """)
    
    visits = cursor.fetchall()
    return visits

@app.get("/visitslist")
def visits(request: Request):    
    participants = read_participants()
    visits = read_visits()
    return templates.TemplateResponse("visitslist.html", {"request": request, "participants": participants, "visits": visits})

@app.post("/submit-visit")
async def submit_visit(selected_participant: int = Form(...), visit_date: date = Form(...), next_surgery: str = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()
    cursor.execute(
    """
    INSERT INTO visits (participant_id, date, next_procedure) VALUES (?, ?, ?);
    """, (selected_participant, visit_date, next_surgery)
    )

    connection.commit()

    return RedirectResponse(url="/visitslist", status_code=303)

def read_experiments():
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM experiments
                   """)
    
    experiments = cursor.fetchall()
    return experiments

def read_table(table_name):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(f"SELECT * FROM {table_name}")
    
    table = cursor.fetchall()
    return table

@app.get("/experimentslist")
def experiemnts(request: Request):
    visits = read_visits()
    # experiments = read_experiments()

    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
                   SELECT *
                   FROM experiments
                   """)
    
    experiments = cursor.fetchall()

    column_names = [description[0] for description in cursor.description]
    experiment_ids = pd.DataFrame(experiments, columns=column_names)['id'].tolist()
    experiment_ids.sort()  # id is 1, 2, 3...

    number_rounds = []
    for experiment_id in experiment_ids:
        cursor.execute(f"SELECT * FROM trials WHERE experiment_id = {experiment_id}")
        number_rounds.append(len(cursor.fetchall()))

    parameters = read_table('parameters') # todo: use this function instead of defining repeats
    return templates.TemplateResponse("experimentslist.html", {"request": request, "visits": visits, "experiments": experiments, "parameters": parameters, "number_rounds": number_rounds})

@app.post("/submit-experiment")
async def submit_experiment(selected_visit: int = Form(...), selected_parameter: int = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
    """
    INSERT INTO experiments (visit_id, parameter_id, round_count) VALUES (?, ?, ?);
    """, (selected_visit, selected_parameter, 0)
    )

    connection.commit()

    return RedirectResponse(url="/experimentslist", status_code=303)


@app.get('/trial')
# @app.get('/trial?selected_experiment={selected_experiment}&round_count={round_count}')
# def trial(request: Request):
# http://127.0.0.1:8000/trial?selected_experiment=1&round_count=3
def trial(request: Request, selected_experiment: int = Query(...)):
# def trial(request: Request, round_count: int = Query(...), selected_experiment: int = Query(...)):
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
async def submit_trial(selected_image: str = Form(...), img1: str = Form(...), img2: str = Form(...), selected_experiment: int = Form(...)):
    # get experiment id, round_count
    experiment_id = selected_experiment#1
    # todo: a function, get round_count by experiment_id
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

    return RedirectResponse(url=f"/trial?selected_experiment={selected_experiment}", status_code=303)
# todo: submit trial goes to http://127.0.0.1:8000/trial, but should go to http://127.0.0.1:8000/trial?selected_experiment=1, submit_trial needs to know selected_experiment

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

