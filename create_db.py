import sqlite3, config

connection = sqlite3.connect(config.DB_FILE)

cursor = connection.cursor()

# previous test
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS patient (
# id INTEGER PRIMARY KEY,
# number INTEGER NOT NULL UNIQUE,
# age INTEGER NOT NULL,
# race TEXT NOT NULL,
# ethnicity TEXT NOT NULL
# );
# """
# )

# cursor.execute(
# """
# INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
# """, (1, 54, 'White', 'Non-Hispanic')
# )

# cursor.execute(
# """
# INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
# """, (2, 51, 'Black', 'Non-Hispanic')
# )

# cursor.execute(
# """
# INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
# """, (3, 58, 'Asian', 'Non-Hispanic')
# )

# patients
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS patients (
id INTEGER PRIMARY KEY,
number INTEGER NOT NULL UNIQUE,
language TEXT NOT NULL
);
"""
)

cursor.execute(
"""
INSERT INTO patients (number, language) VALUES (?, ?);
""", (1, 'English')
)

# providers
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS providers (
id INTEGER PRIMARY KEY,
number INTEGER NOT NULL UNIQUE,
name TEXT NOT NULL
);
"""
)

cursor.execute(
"""
INSERT INTO providers (number, name) VALUES (?, ?);
""", (1, 'Wang')
)

# participants
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS participants (
id INTEGER PRIMARY KEY,
type TEXT NOT NULL,
patient_id INTEGER NOT NULL,
provider_id INTEGER NOT NULL,
FOREIGN KEY(patient_id) REFERENCES patients(id),
FOREIGN KEY(provider_id) REFERENCES providers(id),
UNIQUE (type, patient_id, provider_id)
);
"""
)

cursor.execute(
"""
INSERT INTO participants (type, patient_id, provider_id) VALUES (?, ?, ?);
""", ('provider', 1, 1)
)

cursor.execute(
"""
INSERT INTO participants (type, patient_id, provider_id) VALUES (?, ?, ?);
""", ('patient', 1, 1)
)

# visits
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS visits (
id INTEGER PRIMARY KEY,
participant_id INTEGER NOT NULL,
date DATE NOT NULL,
next_procedure TEXT NOT NULL,
FOREIGN KEY(participant_id) REFERENCES participants(id)
);
"""
)

cursor.execute(
"""
INSERT INTO visits (participant_id, date, next_procedure) VALUES (?, ?, ?);
""", (1, '2023-09-27', 'nipple reconstruction')
)

# experiments
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS experiments (
id INTEGER PRIMARY KEY,
visit_id INTEGER NOT NULL,
parameter_id INTEGER NOT NULL,
FOREIGN KEY(visit_id) REFERENCES visits(id)
);
"""
)

cursor.execute(
"""
INSERT INTO experiments (visit_id, parameter_id) VALUES (?, ?);
""", (1, 1)
)

# perceptual_map
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS perceptual_map (
id INTEGER PRIMARY KEY,
imgdb_img_id INTEGER NOT NULL UNIQUE,
x FLOAT NOT NULL,
y FLOAT NOT NULL
);
"""
)

# import pandas as pd
# import os

# perceptual_map = pd.read_csv(os.path.join(os.getcwd(), 'data/best_embedding_2023-08-31.csv'), header=None)
# for index, row in perceptual_map.iterrows():
#     cursor.execute(
#     """
#     INSERT INTO perceptual_map (imgdb_img_id, x, y) VALUES (?, ?, ?);
#     """, (index, row[0], row[1])
#     )

# imgdb
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS imgdb (
id INTEGER PRIMARY KEY,
img_id INTEGER NOT NULL UNIQUE,
img_name TEXT NOT NULL,
participant_number INTEGER NOT NULL,
new_race TEXT NOT NULL,
age INTEGER NOT NULL,
bmi FLOAT NOT NULL,
days_since_recon INTEGER NOT NULL,
breast_state TEXT NOT NULL
);
"""
)

# import pandas as pd
# # read data
# img_num_map = pd.read_csv("./data/img_num_map.csv").query('img_id < 10000')

# # extract participant number, only need first 3 numbers
# img_num_map_participant_number = img_num_map['img_name'].str.split('_', expand=True).iloc[:, 2].str[:3].astype(int)
# img_num_map['img_name_participant'] = img_num_map_participant_number

# img_base = pd.read_csv("./data/20220801_process_checkpoint_ps.csv")

# # breast state cluster
# breast_states_cluster = (pd.read_csv("./data/breast_states_clusters.csv").loc[:, ['ImageFrom', 'cluster']].
#                          rename(columns={'cluster': 'BREAST_STATE_CLUSTER'}))

# # merge df, drop columns, change column types
# df_master = (img_base
#              .pipe(pd.merge, img_num_map, how='left', left_on='PARTICIPANT_NUMBER', right_on='img_name_participant')
#              .pipe(pd.merge, breast_states_cluster, how='left', left_on='BREAST_STATE', right_on='ImageFrom')
#              .pipe(pd.DataFrame.drop, columns=["FILE_NAME", "IMAGE_ID", "DOWNLOAD", "PROCESS", "N_3D_IMAGES",
#                                                "VISIT_NUMBER", "IMG_ID", "SURGEON_MARKING", "FILLHOLES", "TATTOO",
#                                                "NEED_PS", "PS_DATE", "img_name_participant", "img_relative_path", 
#                                                "ImageFrom", "img_path"])
#              .pipe(pd.DataFrame.astype, {'NEW_RACE': 'category', 'BREAST_STATE': 'category',
#                                          'AGE_CATEGORY': 'category', 'BMI_CATEGORY': 'category',
#                                          'DAYS_SINCE_RECON_CATEGORY': 'category', 'BREAST_STATE_CLUSTER': 'category'})
#              )

# df = df_master.copy()

# for index, row in df.iterrows():
#     cursor.execute(
#     """
#     INSERT INTO imgdb (img_id, img_name, participant_number, new_race, age, bmi, days_since_recon, breast_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
#     """, (row['img_id'], row['img_name'], row['PARTICIPANT_NUMBER'], row['NEW_RACE'], row['AGE'], row['BMI'], row['DAYS_SINCE_RECON'], row['BREAST_STATE'])
#     )

# parameters
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS parameters (
id INTEGER PRIMARY KEY,
algorithm TEXT NOT NULL,
k float NOT NULL,
response_model TEXT NOT NULL,
probability_model TEXT NOT NULL
);
"""
)

cursor.execute(
"""
INSERT INTO parameters (algorithm, k, response_model, probability_model) VALUES (?, ?, ?, ?);
""", ("random pair selection", -1, "na", "na")
)

cursor.execute(
"""
INSERT INTO parameters (algorithm, k, response_model, probability_model) VALUES (?, ?, ?, ?);
""", ("active pair selection", 1.204, "CONSTANT", "BT")
)

cursor.execute(
"""
INSERT INTO parameters (algorithm, k, response_model, probability_model) VALUES (?, ?, ?, ?);
""", ("active pair selection", 5.329, "DECAYING", "BT")
)

############################################
# trials
# cursor.execute(
# """
# CREATE TABLE IF NOT EXISTS trials (
# id INTEGER PRIMARY KEY,
# experiment_id INTEGER NOT NULL,
# round INTEGER NOT NULL,
# img1_id INTEGER NOT NULL,
# img2_id INTEGER NOT NULL,
# select_id INTEGER NOT NULL,
# timepoint TIMESTAMP NOT NULL,
# meanx FLOAT NOT NULL,
# meany FLOAT NOT NULL,
# stdx FLOAT NOT NULL,
# stdy FLOAT NOT NULL,
# FOREIGN KEY(experiment_id) REFERENCES experiments(id)
# );
# """
# )

# cursor.execute(
# """
# INSERT INTO trials (experiment_id, round, img1_id, img2_id, select_id, timepoint, meanx, meany, stdx, stdy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
# """, (1, 1, 2, 3, 2, '2023-04-10 10:39:37', 0.20, -0.23, 0.19, 0.21)
# )

# trials
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS trials (
id INTEGER PRIMARY KEY,
experiment_id INTEGER NOT NULL,
round INTEGER NOT NULL,
img1_id INTEGER NOT NULL,
img2_id INTEGER NOT NULL,
select_id INTEGER NOT NULL,
timepoint TIMESTAMP NOT NULL,
mean TEXT,
cov TEXT,
a TEXT,
tau FLOAT,
FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
"""
)

# import numpy as np
# import json

# mean = np.array([0.08756645, 0.00019115])
# cov = np.array([[0.08257532, -0.00073675], [-0.00073675, 0.08365184]])
# a = np.array([-0.3654387, 0.01344264])
# tau = 0.002649889696594203

# cursor.execute(
# """
# INSERT INTO trials (experiment_id, round, img1_id, img2_id, select_id, timepoint, mean, cov, a, tau) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
# """, (1, 1, 2, 3, 2, '2023-04-10 10:39:37', json.dumps(mean.tolist()), json.dumps(cov.tolist()), json.dumps(a.tolist()), tau)
# )

connection.commit()