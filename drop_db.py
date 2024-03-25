import sqlite3, config


connection = sqlite3.connect(config.DB_FILE)

cursor = connection.cursor()

cursor.execute("""DROP TABLE patients""")
cursor.execute("""DROP TABLE providers""")
cursor.execute("""DROP TABLE participants""")
cursor.execute("""DROP TABLE visits""")
cursor.execute("""DROP TABLE experiments""")

cursor.execute("""DROP TABLE perceptual_map""")
cursor.execute("""DROP TABLE imgdb""")

cursor.execute("""DROP TABLE parameters""")
cursor.execute("""DROP TABLE trials""")
cursor.execute("""DROP TABLE validities""")

connection.commit()
