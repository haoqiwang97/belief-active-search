# uvicorn main:app --reload 
import uvicorn

import sqlite3, config

from fastapi import FastAPI, Request, Form, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse

import logging
from datetime import datetime
from datetime import date  # Import the date type

import pandas as pd

from typing import Tuple, Optional


app = FastAPI()
templates = Jinja2Templates(directory="templates")

img_mount_path = "/img_database_2d"
app.mount(img_mount_path, StaticFiles(directory="./img_database_2d"), name="img_database_2d")
app.mount("/temporary", StaticFiles(directory="./temporary"), name="temporary")

def read_db(table_name: str):
    # read as database format
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    table_db = cursor.fetchall()
    return table_db

def read_pd(table_name: str) -> pd.DataFrame:
    # read as pandas format
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    table_db = cursor.fetchall()

    column_names = [description[0] for description in cursor.description]
    table_pd = pd.DataFrame(table_db, columns=column_names)
    return table_pd

def random_select() -> Tuple[str, str]:
    df = read_pd('imgdb')

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

@app.get("/participantslist")
def participants(request: Request):    
    patients = read_db('patients')
    providers = read_db('providers')
    participants = read_db('participants')
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

@app.get("/visitslist")
def visits(request: Request):    
    participants = read_db('participants')
    visits = read_db('visits')
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

@app.get("/experimentslist")
def experiemnts(request: Request):
    visits = read_db('visits')

    experiments = read_db('experiments')
    experiments_pd = read_pd('experiments')
    experiment_ids = experiments_pd['id'].tolist()
    experiment_ids.sort()  # id is 1, 2, 3...

    number_rounds_list = []
    for experiment_id in experiment_ids:
        number_rounds_list.append(get_number_rounds(experiment_id))

    parameters = read_db('parameters') # todo: use this function instead of defining repeats
    return templates.TemplateResponse("experimentslist.html", {"request": request, "visits": visits, "experiments": experiments, "parameters": parameters, "number_rounds_list": number_rounds_list})

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


import numpy as np
import ast
import json

class Database():
    def __init__(self, experiment_id: int = None):
        self.experiment_id = experiment_id

        # perceptual map related parameters
        self.embedding = read_pd('perceptual_map')[['x', 'y']].to_numpy()  # select x, y columns
        # self.embedding = np.genfromtxt('best_embedding_2023-08-31.csv', delimiter=',')

        self.N = self.embedding.shape[0]  # # N, number of items, used by ActiveQuery
        self.D = self.embedding.shape[1]  # number of dimensions of perceptual map
        self.bounds = [-0.5, 0.5]  # todo: get this according to perceptual map? may need to update stan model

        # bayes model related parameters, read from database, shared by all
        # self.k = 5.329
        # self.k_normalization = 'DECAYING'
        # self.noise_model = 'BT'
        experiments_df = read_pd('experiments')
        parameter_id = experiments_df.query('id == @experiment_id')['parameter_id'].item()
        
        parameters_df = read_pd('parameters')
        parameters = parameters_df.query('id == @parameter_id')
        self.k = parameters['k'].item()
        self.k_normalization = parameters['response_model'].item()
        self.noise_model = parameters['probability_model'].item()
            
        # self.k = 5.329
        self.Nsamples = 4000  # number of samples

        # active query related parameters
        self.Npairs = 50  # random select how many pairs

        # number of rounds done
        number_rounds = get_number_rounds(experiment_id)
        if number_rounds == 0:
            self._start()  # activequery
        else:
            # read database
            self._continue()  # activequery

    def _start(self):
        # if start new, initialization
        self.A = []  # list
        self.tau = []  # numpy list
        self.y_vec = []  # list
        self.response = []

        self.W_samples = np.random.uniform(self.bounds[0], self.bounds[1], (self.Nsamples, self.D))
        
        self.mu_W = np.mean(self.W_samples, 0)
        self.Wcov = np.cov(self.W_samples, rowvar=False)  # todo: just give a number?
        self.A_sel = 0  # self.A[-1]
        self.tau_sel = 0  # self.tau[-1]

    def _continue(self):
        # if continue, read database
        trials = read_pd('trials').query('experiment_id == @self.experiment_id') # todo: change to trials
        
        self.A = [np.array(ast.literal_eval(s)) for s in trials['a'].tolist()]  # list
        self.tau = [np.float64(s) for s in trials['tau'].tolist()]  # numpy list
        self.y_vec = list(np.where(trials['select_id'] == trials['img1_id'], 1, 0))  # list
        self.response = trials['select_id'].tolist()

        self.mu_W = [np.array(ast.literal_eval(s)) for s in trials['mean'].tolist()][-1]
        self.Wcov = [np.array(ast.literal_eval(s)) for s in trials['cov'].tolist()][-1]  # only need the last one
        self.A_sel = self.A[-1]
        self.tau_sel = self.tau[-1]
    
    def initial_estimate(self) -> dict:
        estimation = {'mean': self.mu_W, 'a': self.A_sel, 'tau': self.tau_sel, 'cov': self.Wcov}
        return estimation
    
    def update(self, estimation: dict, query: Tuple[int, int], response: int):
        self.mu_W = estimation['mean']
        self.Wcov = estimation['cov']

        self.A_sel = estimation['a']
        self.tau_sel = estimation['tau']
        self.A.append(self.A_sel)
        self.tau.append(self.tau_sel)

        self.query = query # todo:
        self.response.append(response)
        response_binary = 1 if response == query[0] else 0
        self.y_vec.append(response_binary)

        # write to database
        # query, response
        # 'mean': self.mu_W, 'a': self.A_sel, 'tau': self.tau_sel, 'cov': self.Wcov

        # connection = sqlite3.connect(config.DB_FILE)
        # cursor = connection.cursor()
        # cursor.execute(
        # """
        # INSERT INTO newtrials (experiment_id, round, img1_id, img2_id, select_id, timepoint, mean, cov, a, tau) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        # """, (1, 1, 2, 3, 2, '2023-04-10 10:39:37', json.dumps(mean.tolist()), json.dumps(cov.tolist()), json.dumps(a.tolist()), tau)
        # )

def pair2hyperplane(p, embedding: np.ndarray, normalization: str, slice_point=None):
    # converts pair to hyperplane weights and bias
    A_emb = 2*(embedding[p[0], :] - embedding[p[1], :])  # a, 2(p-q)

    if slice_point is None:
        tau_emb = (np.linalg.norm(embedding[p[0], :])**2 - np.linalg.norm(embedding[p[1], :])**2)  # b, ||p||^2 - ||q||^2
    else:
        tau_emb = np.dot(A_emb, slice_point)

    if normalization == 'CONSTANT':  # normalization
        pass
    elif normalization == 'NORMALIZED':
        A_mag = np.linalg.norm(A_emb)  # ||a||
        A_emb = A_emb / A_mag
        tau_emb = tau_emb / A_mag
    elif normalization == 'DECAYING':
        A_mag = np.linalg.norm(A_emb)
        A_emb = A_emb * np.exp(-A_mag)
        tau_emb = tau_emb * np.exp(-A_mag)
    return (A_emb, tau_emb)


class ActiveQuery():
    def __init__(self, db: Database, method: str):
        # input, fixed, k, method, normalization, Npairs, lambda_pen_EPMV
        self.db = db

        self.method = method
        self.lambda_pen_MCMV = 1

    def get_next_round(self, Wcov: Optional[np.ndarray] = None):
        # just need the previous one step, no need to know history
        lambda_pen_MCMV = self.lambda_pen_MCMV
        embedding = self.db.embedding
        k_normalization = self.db.k_normalization
        k = self.db.k
        mu_W = self.db.mu_W

        N = self.db.N
        Npairs = self.db.Npairs

        # generate all possible pairs
        # Pairs = self.get_all_pairs(self.N)

        # generate random proposal pairs
        Pairs = self.get_random_pairs(N, Npairs)

        # evaluate proposed pairs
        if self.method == 'RANDOM':
            p = Pairs[0]
        elif self.method == 'MCMV':
            value = np.zeros((Npairs,))

            for j in range(Npairs):
                p = Pairs[j]  # pair j
                (A_emb, tau_emb) = pair2hyperplane(p, embedding, k_normalization)  # get normalized a and b as in paper
                varest = np.dot(A_emb, Wcov).dot(A_emb)  # first part of eta
                distmu = np.abs((np.dot(A_emb, mu_W) - tau_emb)  / np.linalg.norm(A_emb))  # second part of eta

                # choose highest variance, but smallest distance to mean
                value[j] = k * np.sqrt(varest) - lambda_pen_MCMV * distmu

            p = Pairs[np.argmax(value)]
        
        query = p

        print(f"Show pair {query}")
        return query
    
    def get_random_pairs(self, N, M):
        # pair selection support function
        # N: sample how many pairs
        # M: Npairs
        indices = np.random.choice(N, (int(1.5*M), 2))
        indices = [(i[0], i[1]) for i in indices if i[0] != i[1]]
        assert len(indices) >= M
        return indices[0:M]

    def get_all_pairs(self, N):
        # pair selection support function
        # N: sample how many pairs
        import itertools
        indices = list(itertools.combinations(range(N), 2))
        return indices
    

import matplotlib.pyplot as plt
import pickle
import pystan


class BayesEstimate():
    stan_model = """
    data {
        int<lower=0> D;       // space dimension
        int<lower=0> M;       // number of measurements so far
        real k;               // logistic noise parameter (scale)
        vector[2] bounds;      // hypercube bounds [lower,upper]
        int y[M];             // measurement outcomes
        vector[D] A[M];       // hyperplane directions
        vector[M] tau;        // hyperplane offsets
    }
    parameters {
        vector<lower=bounds[1],upper=bounds[2]>[D] W;         // the user point
    }
    transformed parameters {
        vector[M] z;
        for (i in 1:M)
            z[i] = dot_product(A[i], W) - tau[i];
    }
    model {
        // prior
        W ~ uniform(bounds[1],bounds[2]);
    
        // linking observations
        y ~ bernoulli_logit(k * z);
    }
    """

    def __init__(self, db: Database):   
        # make model
        try:
            # load Stan model
            self.sm = pickle.load(open('model.pkl', 'rb'))
            print("Loaded saved model")
        except:
            self.sm = pystan.StanModel(model_code=self.stan_model)
            with open('model.pkl', 'wb') as f:
                pickle.dump(self.sm, f)

        self.db = db  # input, read from database, A, tau, y_vec

        # parameters used for bayes sampling
        self.Nchains = 4  # number of chains to sample
        self.Niter = int(2*self.db.Nsamples/self.Nchains)  # number of iterations   

    def fit(self, query: Tuple[int, int], response: int) -> dict:
        embedding = self.db.embedding
        k_normalization = self.db.k_normalization

        (A_sel, tau_sel) = pair2hyperplane(query, embedding, k_normalization)

        A = self.db.A + [A_sel]  # self.db.A.append(A_sel)
        tau = np.append(self.db.tau, tau_sel)

        response_binary = 1 if response == query[0] else 0
        y = response_binary
        y_vec = self.db.y_vec + [y]

        D = self.db.D
        k = self.db.k
        bounds = self.db.bounds
        
        # given measurements 0..i, get posterior samples
        data_gen = {'D': D,  # number of dimensions
                    'k': k,  # noise constant
                    'M': len(A),  # number of measurements so far
                    'A': A, 
                    'tau': tau,
                    'y': y_vec,
                    'bounds': bounds}

        # get posterior samples
        print('Start fitting...')

        fit = self.sm.sampling(data=data_gen, iter=self.Niter, chains=self.Nchains, init=0, n_jobs=1)
        W_samples = fit.extract()['W']

        self.W_samples = W_samples
        self.mu_W = np.mean(W_samples, 0)
        print(f"Current estimate: {self.mu_W}")
        self.Wcov = np.cov(self.W_samples, rowvar=False)  # get covariance

        self.A_sel = A_sel
        self.tau_sel = tau_sel

        estimation = {'mean': self.mu_W, 'a': self.A_sel, 'tau': self.tau_sel, 'cov': self.Wcov}
        return estimation

    def plot(self, query: Tuple[int, int], response: int, user=None):
        W_samples = self.W_samples
        A_sel = self.A_sel#self.A[-1]
        tau_sel = self.tau_sel#self.tau[-1]
        embedding = self.db.embedding
        
        plt.figure(189)
        plt.clf()
        plt.axis([-0.6, 0.6, -0.6, 0.6])
        plt.gca().set_aspect('equal')

        # split left or right side
        Nsplit = 0
        Isplit = []
        for j in range(1, W_samples.shape[0]):
            z = np.dot(A_sel, W_samples[j, :]) - tau_sel
            if z > 0:
                Isplit.append(j)
                Nsplit += 1

        # plot estimate
        estimate_point = self.mu_W
        plt.scatter(estimate_point[0], estimate_point[1], s=50, c='r', marker='x', zorder=2)
        # plot posterior
        plt.plot(W_samples[:, 0], W_samples[:, 1], 'y.', alpha=0.05, zorder=1)
        # plot posterior other half different color
        plt.plot(W_samples[Isplit, 0], W_samples[Isplit, 1], 'c.', alpha=0.05, zorder=1)
        # plot groundtruth
        if user is not None and user.ideal_point is not None:
            plt.plot(user.ideal_point[0], user.ideal_point[1], 'b*')
        # plot landmarks
        plt.scatter(embedding[:, 0], embedding[:, 1], s=25, c='black', zorder=2)
        # plot pair shown this round
        plt.scatter(embedding[query[0], 0], embedding[query[0], 1], s=25, c='m', zorder=2)
        plt.scatter(embedding[query[1], 0], embedding[query[1], 1], s=25, c='m', zorder=2)
        # plot response
        # plt.scatter(embedding[response, 0], embedding[response, 1], s=80, facecolors='none', edgecolors='r', zorder=2)

        # print(f"Current estimate: {estimate_point}")

        plt.ion()

        # save plots
        import os
        folder_name = "temporary"
        plot_name = f"{folder_name}/search.png"
        plt.savefig(plot_name, dpi=300)

        app.mount("/temporary", StaticFiles(directory="./temporary"), name="temporary")


        # plt.pause(self.plot_pause)  # for observation
        # plt.pause(0.5)  # for observation


@app.get('/trial')
# http://127.0.0.1:8000/trial?selected_experiment=1&round_count=3
def trial(request: Request, selected_experiment: int = Query(...)):
    # read image to memory
    # todo: can do it without mount?
    # select 2 images
    # manually select
    img1 = "/img_database_2d/13268_3D_165_6M_121311_UprightHH1_trim_clean_snapshot_noborder.png" # "https://www.w3schools.com/images/picture.jpg" #
    img2 = "/img_database_2d/13589_3D_166v2_6M_110211_UprightHH1_trim_clean_snapshot_noborder.png" # "https://www.w3schools.com/images/w3schools_green.jpg" #
    
    # random select
    img1, img2 = random_select()
    pred = "/temporary/search.png"

    # active select
    # input: need experiment_id, read the database, find what images should be shown/calculate here based on previous database info, but this repeats many variables, I maybe should do all of this in submit-trial.html
    # output: path to 2 images, plot
    # get parameter by selected_experiment
    db = Database(experiment_id=selected_experiment)
    # initialize ActiveQuery based on given parameters
    aq = ActiveQuery(db, 'MCMV')
    # get next round
    estimation = db.initial_estimate()  # maybe no need to do this? no, initialization still need?
    query = aq.get_next_round(estimation['cov'])
    imgdb_pd = read_pd('imgdb')
    img1 = "/img_database_2d/" + imgdb_pd.query('img_id == @query[0]')['img_name'].item()  # todo: item
    img2 = "/img_database_2d/" + imgdb_pd.query('img_id == @query[1]')['img_name'].item()  # todo: item

    # get coordinate
    # app.mount("/temporary", StaticFiles(directory="./temporary"), name="temporary")
    pred = "/temporary/search.png"
    return templates.TemplateResponse("trial.html", {"request": request, "img1": img1, "img2": img2, "pred": pred, 
                                                     "mean": np.around(db.mu_W, decimals=3), 
                                                     "cov": np.around(db.Wcov, decimals=3)})

def get_number_rounds(experiment_id: int):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM trials WHERE experiment_id = {experiment_id}")
    return len(cursor.fetchall())


@app.post("/submit-trial")
async def submit_trial(selected_image: str = Form(...), img1: str = Form(...), img2: str = Form(...), selected_experiment: int = Form(...)):
    # get experiment_id
    experiment_id = selected_experiment
    # get round_count by experiment_id
    round = get_number_rounds(experiment_id)

    img1 = img1[len(img_mount_path)+1: ]
    img2 = img2[len(img_mount_path)+1: ]

    # name to img_id
    df = read_pd('imgdb')
    img1_id = df.loc[df['img_name'] == img1, 'img_id'].item()
    img2_id = df.loc[df['img_name'] == img2, 'img_id'].item()
 
    if selected_image == "img1left":
        print("Button clicked: img1left")
        select_id = img1_id
    else:
        print("Button clicked: img2right")
        select_id = img2_id

    # estimate user point
    db = Database(experiment_id)
    be = BayesEstimate(db)
    # input: previous selections
    # output: figure, mean, std
    # do not show the same pair appeared before
    # write to database
    query = (img1_id, img2_id)
    response = selected_image
    
    estimation = be.fit(query, response)
    be.plot(query, response)

    # db.update(estimation, query, response)
    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()
    cursor.execute(
    """
    INSERT INTO trials (experiment_id, round, img1_id, img2_id, select_id, timepoint, mean, cov, a, tau) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (experiment_id, round+1, img1_id, img2_id, select_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), json.dumps(estimation['mean'].tolist()), json.dumps(estimation['cov'].tolist()), json.dumps(estimation['a'].tolist()), estimation['tau'])
    )

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)