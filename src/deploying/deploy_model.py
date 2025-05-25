import os

def deploy():
    print("🚀 Deploying model API using Flask...")
    os.system("python app/main.py")  # or use subprocess.run

if __name__ == "__main__":
    deploy()