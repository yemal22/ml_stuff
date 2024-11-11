import psycopg2
import psycopg2.extras

hostname = 'localhost'
username = 'postgres'
database = 'demo'
pwd = 'kemykemy'
port = '5432'

conn = None

try:
    with psycopg2.connect(
        host=hostname,
        user=username,
        password=pwd,
        dbname=database,
        port=port
    ) as conn:
        print("Connected to database")
        print("-"*20)

        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:

            cur.execute('DROP TABLE IF EXISTS employee')

            create_script = ''' CREATE TABLE IF NOT EXISTS employee (
                                    id      int PRIMARY KEY,
                                    name    varchar(40) NOT NULL,
                                    salary  int,
                                    dept_id varchar(30))'''
            
            cur.execute(create_script)
            print("Table created")
            print("-"*20)

            insert_script = 'INSERT INTO employee (id, name, salary, dept_id) VALUES (%s, %s, %s, %s)'
            insert_value = [(1, 'James', 12000, 'D1'), (2, 'John', 37000, 'D2'), (3, 'Xavier', 15000, 'D1')]

            for record in insert_value:    
                cur.execute(insert_script, record)
            print(f'{len(insert_value)} insertion(s) done')
            print("-"*20)

            cur.execute('SELECT * FROM employee')
            rows = cur.fetchall()
            print("Data in employee table:")
            print("-"*20)
            for row in rows:
                print(row['name'], row['salary'])

except Exception as error:
    print("Error: ", error)

finally:
    if conn is not None:
        conn.close()
        print("connexion close")
        print("-"*20)