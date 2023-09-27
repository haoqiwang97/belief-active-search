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

# cursor.execute(
# """
# INSERT INTO patients (number, language) VALUES (?, ?);
# """, (1, 'English')
# )

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

# cursor.execute(
# """
# INSERT INTO providers (number, name) VALUES (?, ?);
# """, (1, 'noname')
# )

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

# cursor.execute(
# """
# INSERT INTO participants (type, patient_id, provider_id) VALUES (?, ?, ?);
# """, ('provider', 1, 1)
# )

# cursor.execute(
# """
# INSERT INTO participants (type, patient_id, provider_id) VALUES (?, ?, ?);
# """, ('patient', 1, 1)
# )

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

# cursor.execute(
# """
# INSERT INTO visits (participant_id, date, next_procedure) VALUES (?, ?, ?);
# """, (1, '2023-09-27', 'nipple reconstruction')
# )

# experiments
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS experiments (
id INTEGER PRIMARY KEY,
visit_id INTEGER NOT NULL,
parameter_id INTEGER NOT NULL,
round_count INTEGER NOT NULL,
FOREIGN KEY(visit_id) REFERENCES visits(id)
);
"""
)

# cursor.execute(
# """
# INSERT INTO experiments (visit_id, parameter_id, round_count) VALUES (?, ?, ?);
# """, (1, 1, 0)
# )

# trials
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS trials (
id INTEGER PRIMARY KEY,
experiment_id INTEGER NOT NULL,
round INTEGER NOT NULL,
img1 INTEGER NOT NULL,
img2 INTEGER NOT NULL,
selection INTEGER NOT NULL,
timepoint TIMESTAMP NOT NULL,
meanx FLOAT NOT NULL,
meany FLOAT NOT NULL,
stdx FLOAT NOT NULL,
stdy FLOAT NOT NULL,
FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
"""
)

# cursor.execute(
# """
# INSERT INTO trials (experiment_id, round, img1, img2, selection, timepoint, meanx, meany, stdx, stdy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
# """, (1, 1, 2, 3, 2, '2023-04-10 10:39:37', 0.20, -0.23, 0.19, 0.21)
# )

connection.commit()