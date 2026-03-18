import os
import pickle

# Hardcoded secret (BLOCKER)
API_KEY = "sk-12345-super-secret-key"
DB_PASSWORD = "admin123"

def run_user_command(user_input):
    # Command injection (BLOCKER)
    os.system(f"echo {user_input}")
    result = eval(user_input)
    return result

def get_users(db):
    # N+1 query pattern (WARNING)
    users = db.query("SELECT * FROM users")
    for user in users:
        orders = db.query(f"SELECT * FROM orders WHERE user_id = {user['id']}")
        user['orders'] = orders
    return users

def load_data(raw_bytes):
    # Unsafe deserialization (BLOCKER)
    return pickle.loads(raw_bytes)
