import os
import pickle

API_KEY = "sk-12345-super-secret-key"
DB_PASSWORD = "admin123"

def run_user_command(user_input):
    os.system(f"echo {user_input}")
    result = eval(user_input)
    return result
    
def get_users(db):
    users = db.query("SELECT * FROM users")
    for user in users:
        orders = db.query(f"SELECT * FROM orders WHERE user_id = {user['id']}")
        user['orders'] = orders
    return users
