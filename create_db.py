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

cursor.execute(
"""
INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
""", (1, 54, 'White', 'Non-Hispanic')
)

cursor.execute(
"""
INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
""", (2, 51, 'Black', 'Non-Hispanic')
)

cursor.execute(
"""
INSERT INTO patient (number, age, race, ethnicity) VALUES (?, ?, ?, ?);
""", (3, 58, 'Asian', 'Non-Hispanic')
)

connection.commit()