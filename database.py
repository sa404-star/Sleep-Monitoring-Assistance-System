
import sqlite3
import hashlib
from pathlib import Path

def get_db_path():
    return str(Path(__file__).parent / 'users.db')

def init_db():
    conn = sqlite3.connect(get_db_path())
    c = conn.cursor()
    
    # 创建用户表
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'normal_user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, role='normal_user'):
    try:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        
        hashed_password = hash_password(password)
        c.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                  (username, hashed_password, role))
        
        conn.commit()
        return True, '注册成功'
    except sqlite3.IntegrityError:
        return False, '用户名已存在'
    except Exception as e:
        return False, f'注册失败: {str(e)}'
    finally:
        conn.close()

def verify_user(username, password):
    try:
        conn = sqlite3.connect(get_db_path())
        c = conn.cursor()
        
        hashed_password = hash_password(password)
        c.execute('SELECT id, role FROM users WHERE username = ? AND password = ?',
                  (username, hashed_password))
        
        result = c.fetchone()
        if result:
            return True, {'user_id': result[0], 'role': result[1]}
        return False, '用户名或密码错误'
    except Exception as e:
        return False, f'验证失败: {str(e)}'
    finally:
        conn.close()

# 初始化数据库
init_db()