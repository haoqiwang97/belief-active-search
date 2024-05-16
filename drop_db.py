import sqlite3, config


connection = sqlite3.connect(config.DB_FILE)

cursor = connection.cursor()

# 1 patients
cursor.execute("""DROP TABLE patients""")
# 2 providers
cursor.execute("""DROP TABLE providers""")
# 3 participants
cursor.execute("""DROP TABLE participants""")
# 4 visits
cursor.execute("""DROP TABLE visits""")
# 5 experiments
cursor.execute("""DROP TABLE experiments""")

# 6 perceptual_map
cursor.execute("""DROP TABLE perceptual_map""")
# 7 imgdb
cursor.execute("""DROP TABLE imgdb""")

# 8 parameters
cursor.execute("""DROP TABLE parameters""")
# 9 trials
cursor.execute("""DROP TABLE trials""")
# 10 validities
cursor.execute("""DROP TABLE validities""")

connection.commit()
