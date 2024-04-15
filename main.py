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

def read_db_by_experiment(table_name: str, experiment_id: int):
    # read as database format
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name} WHERE experiment_id = {experiment_id}")
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

def read_pd_by_experiment(table_name: str, experiment_id: int) -> pd.DataFrame:
    # read as pandas format
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    cursor.execute(f"SELECT * FROM {table_name} WHERE experiment_id = {experiment_id}")
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

    parameters = read_db('parameters')
    return templates.TemplateResponse("experimentslist.html", {"request": request, "visits": visits, "experiments": experiments, "parameters": parameters, "number_rounds_list": number_rounds_list})

@app.post("/submit-experiment")
async def submit_experiment(selected_visit: int = Form(...), selected_parameter: int = Form(...)):
    connection = sqlite3.connect(config.DB_FILE)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
    """
    INSERT INTO experiments (visit_id, parameter_id) VALUES (?, ?);
    """, (selected_visit, selected_parameter)
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
        self.embedding = read_pd('perceptual_map')[['x', 'y']].to_numpy()  # select x, y columns, # self.embedding = np.genfromtxt('best_embedding_2023-08-31.csv', delimiter=',')

        self.N = self.embedding.shape[0]  # # N, number of items, used by ActiveQuery
        self.D = self.embedding.shape[1]  # number of dimensions of perceptual map
        self.bounds = [-0.5, 0.5]  # todo: get this according to perceptual map? may need to update stan model

        # bayes model related parameters, read from database, shared by all
        experiments_df = read_pd('experiments')
        parameter_id = experiments_df.query('id == @experiment_id')['parameter_id'].item()
        
        parameters_df = read_pd('parameters')
        parameters = parameters_df.query('id == @parameter_id')
        self.k = parameters['k'].item()  # self.k = 5.329
        self.k_normalization = parameters['response_model'].item()  # self.k_normalization = 'DECAYING'
        self.noise_model = parameters['probability_model'].item()  # self.noise_model = 'BT'
        
        # todo: method, random/mcmv
        self.Nsamples = 4000  # number of samples

        # active query related parameters
        self.Npairs = 50  # random select how many pairs

        # number of rounds done
        number_rounds = get_number_rounds(experiment_id)
        self.number_rounds = number_rounds
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
        self.Wcov = np.cov(self.W_samples, rowvar=False)
        self.A_sel = 0  # self.A[-1]
        self.tau_sel = 0  # self.tau[-1]

    def _continue(self):
        # if continue, read database
        trials = read_pd('trials').query('experiment_id == @self.experiment_id')
        
        self.A = [np.array(ast.literal_eval(s)) for s in trials['a'].tolist()]  # list
        self.tau = [np.float64(s) for s in trials['tau'].tolist()]  # numpy list
        self.y_vec = list(np.where(trials['select_id'] == trials['img1_id'], 1, 0))  # list
        self.response = trials['select_id'].tolist()

        self.mu_W_list = [np.array(ast.literal_eval(s)) for s in trials['mean'].tolist()]
        self.mu_W = [np.array(ast.literal_eval(s)) for s in trials['mean'].tolist()][-1]
        self.Wcov = [np.array(ast.literal_eval(s)) for s in trials['cov'].tolist()][-1]  # only need the last one
        self.A_sel = self.A[-1]
        self.tau_sel = self.tau[-1]
    
    def latest_estimate(self) -> dict:
        estimation = {'mean': self.mu_W, 'a': self.A_sel, 'tau': self.tau_sel, 'cov': self.Wcov}
        return estimation
    
    def previous_mean(self, number_rounds_previous: int = 10):
        trials = read_pd('trials').query('experiment_id == @self.experiment_id')
        return [np.array(ast.literal_eval(s)) for s in trials['mean'].tolist()][int(-1 * number_rounds_previous)]
    

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
        # try:
        # load Stan model
        self.sm = pickle.load(open('model.pkl', 'rb'))
        print("Loaded saved model")
        # except:
        #     self.sm = pystan.StanModel(model_code=self.stan_model)
        #     with open('model.pkl', 'wb') as f:
        #         pickle.dump(self.sm, f)

        self.db = db  # input, read from database, A, tau, y_vec

        # parameters used for bayes sampling
        self.Nchains = 4  # number of chains to sample
        self.Niter = int(2*self.db.Nsamples/self.Nchains)  # number of iterations   

    def fit(self, query: Optional[Tuple[int, int]] = None, response: Optional[int] = None) -> dict:
        embedding = self.db.embedding
        k_normalization = self.db.k_normalization
        D = self.db.D
        k = self.db.k
        bounds = self.db.bounds

        if not (query == None and response == None):
            print("New query")
            (A_sel, tau_sel) = pair2hyperplane(query, embedding, k_normalization)

            A = self.db.A + [A_sel]  # self.db.A.append(A_sel)
            tau = np.append(self.db.tau, tau_sel)

            response_binary = 1 if response == query[0] else 0
            y = response_binary
            y_vec = self.db.y_vec + [y]

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
        
        else:
            print("Not new query")
            A = self.db.A
            tau = self.db.tau
            y_vec = self.db.y_vec

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

            self.A_sel = self.db.A_sel
            self.tau_sel = self.db.tau_sel

            estimation = {'mean': self.mu_W, 'a': self.A_sel, 'tau': self.tau_sel, 'cov': self.Wcov}
            return estimation

    def plot(self, query: Optional[Tuple[int, int]] = None, response: Optional[int] = None, user=None):
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
        if not query == None:
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

        # plt.pause(self.plot_pause)  # for observation
        # plt.pause(0.5)  # for observation

import time
# Function to generate a timestamp
# To dynamically update the image path in your HTML template when a new image is generated or updated, you should include a version or timestamp in the image URL. 
# This way, when a new image is created or updated, the URL changes, forcing the browser to request the updated image from the server.
def get_timestamp():
    return int(time.time())

from scipy.spatial.distance import cdist

@app.get('/trial')
# http://127.0.0.1:8000/trial?selected_experiment=1
def trial(request: Request, selected_experiment: int = Query(...)):
    # read image to memory
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
    # if db.number_rounds > 10 and db.number_rounds % 10 == 0: # todo: if not done this
    #     return templates.TemplateResponse("validity.html", {"request": request})
    
    # initialize ActiveQuery based on given parameters
    aq = ActiveQuery(db, 'MCMV')  # todo: method, random/mcmv
    # get next round
    estimation = db.latest_estimate()  # maybe no need to do this? no, initialization still need?
    query = aq.get_next_round(estimation['cov'])
    imgdb_pd = read_pd('imgdb')
    img1 = "/img_database_2d/" + imgdb_pd.query('img_id == @query[0]')['img_name'].item()
    img2 = "/img_database_2d/" + imgdb_pd.query('img_id == @query[1]')['img_name'].item()

    # get prediction, i.e. posterior distribution plot
    # todo: spanish version not ready yet
    timestamp = get_timestamp()
    pred = f"/temporary/search.png?timestamp={timestamp}"
    # todo: print current coordinate twice
    # get coordinate
    # find point that is closest to the estimation['mean']
    distances = cdist([estimation['mean']], db.embedding)  # make 2d array
    # read perceptual map, find image id
    closest_neighbor_img_id = np.argmin(distances)
    # find path of image id
    closest_neighbor_img = "/img_database_2d/" + imgdb_pd.query('img_id == @closest_neighbor_img_id')['img_name'].item()


    return templates.TemplateResponse("trial.html", {"request": request, "img1": img1, "img2": img2, "pred": pred, 
                                                     "closest_neighbor_img": closest_neighbor_img,
                                                     "number_rounds": db.number_rounds,
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

    img1_short = img1[len(img_mount_path)+1: ]
    img2_short = img2[len(img_mount_path)+1: ]

    # name to img_id
    df = read_pd('imgdb')
    img1_id = df.loc[df['img_name'] == img1_short, 'img_id'].item()
    img2_id = df.loc[df['img_name'] == img2_short, 'img_id'].item()
 
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
    # todo: do not show the same pair appeared before
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

    # if round > 10 and round % 3 == 0: # todo: if not done this, go to validity form and then go to trial
    #     return RedirectResponse(url=f"/validity?selected_experiment={selected_experiment}&round={round}", status_code=303) # give experiment id, round
    # current approximation and previous approximation

    # if done the above, go to next trial
    return RedirectResponse(url=f"/trial?selected_experiment={selected_experiment}", status_code=303)

@app.get("/", response_class=HTMLResponse)
async def write_home(request: Request):
    # return templates.TemplateResponse("add_patient.html", {"request": request})
    return RedirectResponse(url="/home")

@app.get("/validity")
def validity(request: Request, selected_experiment: int, round: int):
    db = Database(experiment_id=selected_experiment)
    imgdb_pd = read_pd('imgdb')
    # todo: duplicate code
    # todo: this is current one, add previous one
    estimation = db.latest_estimate()
    distances = cdist([estimation['mean']], db.embedding)  # make 2d array
    closest_neighbor_img_id = np.argmin(distances)
    closest_neighbor_img = "/img_database_2d/" + imgdb_pd.query('img_id == @closest_neighbor_img_id')['img_name'].item()

    # previous one
    distances = cdist([db.previous_mean()], db.embedding)  # make 2d array
    closest_neighbor_img_id = np.argmin(distances)
    closest_neighbor_img_prev = "/img_database_2d/" + imgdb_pd.query('img_id == @closest_neighbor_img_id')['img_name'].item()
    # todo: consider if I should include the 3rd question
    return templates.TemplateResponse("validity.html", {"request": request, "closest_neighbor_img": closest_neighbor_img, "closest_neighbor_img_prev": closest_neighbor_img_prev})

@app.post("/submit-validity")
async def submit_validity(q1: str = Form(...), q2: str = Form(...), selected_experiment: int = Form(...), round: int = Form(...), closest_neighbor_img: str = Form(...)):
# async def submit_validity(q1: str = Form(...), q2: str = Form(...), q3: str = Form(...), selected_experiment: int = Form(...), round: int = Form(...)):
    # if q3 == 'left':
    #     pick_current = True
    # else:
    #     pick_current = False
    print(selected_experiment, round, closest_neighbor_img, q1, q2) # print(q1, q2, pick_current, selected_experiment, round)

    img_short = closest_neighbor_img[len(img_mount_path)+1: ]
    df = read_pd('imgdb')
    img_id = df.loc[df['img_name'] == img_short, 'img_id'].item()

    connection = sqlite3.connect(config.DB_FILE)
    cursor = connection.cursor()
    cursor.execute(
    """
    INSERT INTO validities (experiment_id, round, top_rank_img, score, doctor_understand) VALUES (?, ?, ?, ?, ?);
    """, (selected_experiment, round, img_id, int(q1), q2)
    )

    connection.commit()

    return RedirectResponse(url=f"/trial?selected_experiment={selected_experiment}", status_code=303)

@app.get("/result")
def result(request: Request, selected_experiment: int = Query(...)):
    # prepare
    experiment_id = selected_experiment
    db = Database(experiment_id=selected_experiment)
    plt.switch_backend('Agg')

    trials = read_db_by_experiment("trials", experiment_id)
    validities = read_db_by_experiment("validities", experiment_id)

    # trial plot
    # read trial
    mean = np.concatenate(db.mu_W_list).reshape(-1, 2)
    fig, ax = plt.subplots()
    embedding = db.embedding
    ax.scatter(embedding[:, 0], embedding[:, 1], s=25, c='black', alpha=0.5)
    ax.quiver(mean[:-1, 0], mean[:-1, 1], mean[1:, 0]-mean[:-1, 0], mean[1:, 1]-mean[:-1, 1], scale_units='xy', angles='xy', scale=1, width=.005, color='tab:red')
    ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), aspect='equal')
    fig.savefig('./temporary/trial_summary.png', dpi=300)

    # validity plot
    validities_pd = read_pd("validities").query('experiment_id == @experiment_id')
    fig, ax = plt.subplots()
    ax.plot(validities_pd['round'], validities_pd['score'])
    ax.scatter(validities_pd['round'], validities_pd['score'], 
               c=['tab:red' if value == 'Yes' else 'tab:blue' for value in list(validities_pd['doctor_understand'])], 
               label='Doctor Understand', zorder=10)
    # Adding legend manually
    legend_dict = {'Yes': 'tab:red', 'No': 'tab:blue'}
    ax.legend(title='Q2', loc='lower right', handles=[plt.Line2D([0], [0], marker='o', color=color, label=label, linestyle='None') for label, color in legend_dict.items()])
    ax.set(ylim=(0, 6), xlabel='Round', ylabel='Q1')
    fig.savefig('./temporary/validity_summary.png', dpi=300)

    # prediction plot
    be = BayesEstimate(db)
    be.fit()
    be.plot()

    # neighbor
    # db = Database(experiment_id=selected_experiment)
    estimation = db.latest_estimate()  # maybe no need to do this? no, initialization still need?
    distances = cdist([estimation['mean']], db.embedding)  # make 2d array
    closest_neighbor_img_ids = distances.squeeze().argsort()[:5]
    imgdb_pd = read_pd('imgdb')
    closest_neighbor_img_list = ["/img_database_2d/" + imgdb_pd.query('img_id == @closest_neighbor_img_id')['img_name'].item() for closest_neighbor_img_id in closest_neighbor_img_ids]
    print(closest_neighbor_img_list)

    # image list
    # imgdb_pd = read_pd('imgdb')
    img_paths = {}
    for index ,row in imgdb_pd.iterrows():
        img_paths[row['img_id']] = "/img_database_2d/" + row['img_name']

    # load plots
    timestamp = get_timestamp()
    trial_plot = f"/temporary/trial_summary.png?timestamp={timestamp}"
    validity_plot = f"/temporary/validity_summary.png?timestamp={timestamp}"
    prediction_plot = f"/temporary/search.png?timestamp={timestamp}"

    return templates.TemplateResponse("result.html", 
                                      {"request": request, 
                                       "trials": trials, 
                                       "validities": validities,
                                       "trial_plot": trial_plot,
                                       "validity_plot": validity_plot,
                                       "prediction_plot": prediction_plot,
                                       "closest_neighbor_img_list": closest_neighbor_img_list,
                                       'img_paths': img_paths,})


@app.get("/satisfaction")
def satisfaction(request: Request, selected_experiment: int = Query(...)):
    # prepare
    experiment_id = selected_experiment
    db = Database(experiment_id=selected_experiment)
    plt.switch_backend('Agg')

    trials = read_db_by_experiment("trials", experiment_id)
    validities = read_db_by_experiment("validities", experiment_id)

    # trial plot
    # read trial
    mean = np.concatenate(db.mu_W_list).reshape(-1, 2)
    fig, ax = plt.subplots()
    embedding = db.embedding
    ax.scatter(embedding[:, 0], embedding[:, 1], s=25, c='black', alpha=0.5)
    ax.quiver(mean[:-1, 0], mean[:-1, 1], mean[1:, 0]-mean[:-1, 0], mean[1:, 1]-mean[:-1, 1], scale_units='xy', angles='xy', scale=1, width=.005, color='tab:red')
    ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), aspect='equal')
    fig.savefig('./temporary/trial_summary.png', dpi=300)

    # validity plot
    validities_pd = read_pd("validities").query('experiment_id == @experiment_id')
    fig, ax = plt.subplots()
    ax.plot(validities_pd['round'], validities_pd['score'])
    ax.scatter(validities_pd['round'], validities_pd['score'], 
               c=['tab:red' if value == 'Yes' else 'tab:blue' for value in list(validities_pd['doctor_understand'])], 
               label='Doctor Understand', zorder=10)
    # Adding legend manually
    legend_dict = {'Yes': 'tab:red', 'No': 'tab:blue'}
    ax.legend(title='Q2', loc='lower right', handles=[plt.Line2D([0], [0], marker='o', color=color, label=label, linestyle='None') for label, color in legend_dict.items()])
    ax.set(ylim=(0, 6), xlabel='Round', ylabel='Q1')
    fig.savefig('./temporary/validity_summary.png', dpi=300)

    # prediction plot
    be = BayesEstimate(db)
    be.fit()
    be.plot()

    # neighbor
    # db = Database(experiment_id=selected_experiment)
    estimation = db.latest_estimate()  # maybe no need to do this? no, initialization still need?
    distances = cdist([estimation['mean']], db.embedding)  # make 2d array
    closest_neighbor_img_ids = distances.squeeze().argsort()[:5]
    imgdb_pd = read_pd('imgdb')
    closest_neighbor_img_list = ["/img_database_2d/" + imgdb_pd.query('img_id == @closest_neighbor_img_id')['img_name'].item() for closest_neighbor_img_id in closest_neighbor_img_ids]
    print(closest_neighbor_img_list)

    # image list
    # imgdb_pd = read_pd('imgdb')
    img_paths = {}
    for index ,row in imgdb_pd.iterrows():
        img_paths[row['img_id']] = "/img_database_2d/" + row['img_name']

    # load plots
    timestamp = get_timestamp()
    trial_plot = f"/temporary/trial_summary.png?timestamp={timestamp}"
    validity_plot = f"/temporary/validity_summary.png?timestamp={timestamp}"
    prediction_plot = f"/temporary/search.png?timestamp={timestamp}"

    return templates.TemplateResponse("satisfaction.html", 
                                      {"request": request, 
                                       "trials": trials, 
                                       "validities": validities,
                                       "trial_plot": trial_plot,
                                       "validity_plot": validity_plot,
                                       "prediction_plot": prediction_plot,
                                       "closest_neighbor_img_list": closest_neighbor_img_list,
                                       'img_paths': img_paths,})


################################################################################
# Create the Dash application, make sure to adjust requests_pathname_prefx
def create_dash_app(dash_url, embedding, mean, img_paths, trials_pd):
    from dash import Dash, dcc, html, Input, Output, callback, no_update
    import plotly.graph_objects as go
    import pandas as pd

    import io
    import base64
    from PIL import Image


    fig = go.Figure(data=[
        go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode="markers",
            marker=dict(
                color='gray',
                opacity=0.8,
            ),
            hoverinfo="none",
            hovertemplate=None,
            showlegend=False  # Disable legend for scatter plot
        )
    ])

    arrow_data = []
    x_end, y_end = 0, 0
    for i in range(len(mean)):
        x_start, y_start = x_end, y_end
        x_end = mean[i, 0]
        y_end = mean[i, 1]
        arrow_data.append({'ID': f'Round {i+1}', 'X_Start': x_start, 'Y_Start': y_start, 'X_End': x_end, 'Y_End': y_end})

    arrows_df = pd.DataFrame(arrow_data)
    trials_pd['timepoint'] = pd.to_datetime(trials_pd['timepoint'])
    trials_pd['relative_time'] = (trials_pd['timepoint'] - trials_pd['timepoint'].iloc[0]).dt.total_seconds()
    trials_pd['relative_time2'] = trials_pd['relative_time'].apply(lambda x: f"{int(x//60)}min{int(x%60)}s")
    arrows_df['relative_time2'] = trials_pd['relative_time2']

    for _, arrow in arrows_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[arrow['X_Start'], arrow['X_End']],
            y=[arrow['Y_Start'], arrow['Y_End']],
            mode='lines+markers',
            line=dict(color='red', width=1),
            marker=dict(symbol='arrow', size=5, angleref='previous'),
            showlegend=False,  # Disable legend for arrows
            name=arrow['ID'],
            hovertemplate=f"{arrow['ID']}"
        ))

    fig.update_layout(
        plot_bgcolor='rgba(255,255,255,0.1)',
        xaxis_title=None, yaxis_title=None,
        xaxis=dict(scaleanchor="y", scaleratio=1,),
        yaxis=dict(scaleanchor="x",scaleratio=1,),
        # margin=dict(l=0, r=0, t=0, b=0),  # Optional: Adjust the margin
    )

    fig.update_xaxes(showticklabels=False) # Hide x axis ticks 
    fig.update_yaxes(showticklabels=False) # Hide y axis ticks

    app_dash = Dash(__name__, requests_pathname_prefix=dash_url+'/') # important to have / at the end

    num_anchors = 5
    step_size = (len(arrows_df) - 1) // (num_anchors - 1)  # Adjust step size calculation
    anchor_points = [1] + [i * step_size + 1 for i in range(1, num_anchors - 1)] + [len(arrows_df)]  # Include first and last points
    print(anchor_points)
    app_dash.layout = html.Div([
        html.H1(children='Interview', style={'textAlign':'center', 'fontFamily': 'sans-serif'}),
        dcc.Graph(id="graph-basic-2", figure=fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
        html.H4(children='Trajectory', style={'textAlign':'center', 'fontFamily': 'sans-serif'}),
        html.Div(
            dcc.RangeSlider(
                id='arrow-range-slider',
                min=1,
                max=len(arrows_df),
                step=1,
                value=[1, len(arrows_df)],
                marks={anchor: f"Time={arrows_df.loc[anchor-1, 'relative_time2']}(Round={anchor})" for anchor in anchor_points}
            ),
            style={'width': '70%', 'margin': '0 auto', 'fontFamily': 'sans-serif'}
        ),
        # html.H4(children='Trial history', style={'textAlign':'center'}) # todo: add history
    ])

    @app_dash.callback(
        [Output("graph-basic-2", "figure"),
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children")],
        [Input("arrow-range-slider", "value"),
        Input("graph-basic-2", "hoverData")]
    )


    def update_figure_and_display_hover(value, hoverData):
        start_id, end_id = value
        filtered_arrows_df = arrows_df.iloc[start_id-1:end_id]

        fig = go.Figure(data=[
            go.Scatter(
                x=embedding[:, 0],
                y=embedding[:, 1],
                mode="markers",
                marker=dict(
                    color='gray',
                    opacity=0.8,
                ),
                hoverinfo="none",
                hovertemplate=None,
                showlegend=False  # Disable legend for scatter plot
            )
        ])
        
        for _, arrow in filtered_arrows_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[arrow['X_Start'], arrow['X_End']],
                y=[arrow['Y_Start'], arrow['Y_End']],
                mode='lines+markers',
                line=dict(color='red', width=1),
                marker=dict(symbol='arrow', size=5, angleref='previous'),
                showlegend=False,  # Disable legend for arrows
                name=arrow['ID'],
                hovertemplate=f"{arrow['ID']}"
            ))

        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.1)',
            xaxis_title=None, yaxis_title=None,
            xaxis=dict(scaleanchor="y", scaleratio=1,),
            yaxis=dict(scaleanchor="x",scaleratio=1,),
            # margin=dict(l=0, r=0, t=0, b=0),  # Optional: Adjust the margin
        )
        fig.update_xaxes(showticklabels=False) # Hide x axis ticks 
        fig.update_yaxes(showticklabels=False) # Hide y axis ticks
        
        if hoverData is None:
            return fig, False, no_update, no_update

        pt = hoverData["points"][0]
        curve_number = pt["curveNumber"]
        if curve_number == 0:
            bbox = pt["bbox"]
            num = pt["pointNumber"]

            img_src = img_paths[num]
            im = Image.open('.' + img_src)
            im = im.convert('RGB')
            # dump it to base64
            buffer = io.BytesIO()
            im.save(buffer, format="jpeg")
            encoded_image = base64.b64encode(buffer.getvalue()).decode()
            im_url = "data:image/jpeg;base64, " + encoded_image

            children = [
                html.Div([
                    html.Img(src=im_url, style={"width": "100%"}),
                    html.H4(f"{num}"),
                ], style={'width': '200px', 'white-space': 'normal'})
            ]
            return fig, True, bbox, children
        else:
            return fig, False, no_update, no_update
    return app_dash

# app_dash = create_dash_app()
################################################################################
from fastapi.middleware.wsgi import WSGIMiddleware


@app.get("/launch-dash")
async def launch_dash(request: Request, selected_experiment: int = Query(...)):
    # prepare
    db = Database(experiment_id=selected_experiment)
    mean = np.concatenate(db.mu_W_list).reshape(-1, 2)
    embedding = db.embedding
    imgdb_pd = read_pd('imgdb')
    trials_pd = read_pd_by_experiment("trials", experiment_id=selected_experiment)
    img_paths = {}
    for index ,row in imgdb_pd.iterrows():
        img_paths[row['img_id']] = "/img_database_2d/" + row['img_name']
    dash_url = f"/dash/{selected_experiment}"
    app_dash = create_dash_app(dash_url, embedding, mean, img_paths, trials_pd)
    # note: if run the same selected_experiment with updated data, it needs like 1 minute to update the old dash app
    # Now mount you dash server into main fastapi application
    app.mount(dash_url, WSGIMiddleware(app_dash.server))
    return RedirectResponse(url=dash_url, status_code=303)

# Now mount you dash server into main fastapi application
# app.mount("/dash", WSGIMiddleware(app_dash.server))
################################################################################


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)