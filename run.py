import os 

def execute_system():
    bash1 = 'python src/01_generate_embedding.py'
    os.system(bash1)

    print('Executed successfully!! Now run app.py')

if __name__ == '__main__':
    execute_system()