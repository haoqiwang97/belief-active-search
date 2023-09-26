import sqlite3, config

connection = sqlite3.connect(config.DB_FILE)

cursor = connection.cursor()

cursor.execute(
"""
CREATE TABLE IF NOT EXISTS patient (
id INTEGER PRIMARY KEY,
number INTEGER NOT NULL UNIQUE,
age INTEGER NOT NULL,
race TEXT NOT NULL,
ethnicity TEXT NOT NULL
);
"""
)

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
stdy FLOAT NOT NULL
);
"""
)

# cursor.execute(
# """
# INSERT INTO trials (experiment_id, round, img1, img2, selection, timepoint, meanx, meany, stdx, stdy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
# """, (1, 1, 2, 3, 2, '2023-04-10 10:39:37', 0.20, -0.23, 0.19, 0.21)
# )

connection.commit()