import sqlite3

conn = sqlite3.connect("databases/resume_rank.db")
cur = conn.cursor()

print("Jobs:")
for row in cur.execute("SELECT * FROM jobs LIMIT 5;"):
    print(row)

print("\nResumes:")
for row in cur.execute("SELECT * FROM resumes LIMIT 5;"):
    print(row)
print("\nrankings full table")
for row in cur.execute("SELECT * FROM rankings LIMIT 5;"):
    print(row)

print("\nRankings:")
for row in cur.execute("""
    SELECT r.filename, ra.final_score
    FROM rankings ra
    JOIN resumes r ON ra.resume_id = r.resume_id
    ORDER BY ra.final_score DESC LIMIT 5;
"""):
    print(row)

conn.close()
