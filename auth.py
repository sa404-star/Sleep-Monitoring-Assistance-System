import sqlite3
import hashlib
from datetime import datetime

class Auth:
    def __init__(self, db_path='users.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库，创建用户表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password):
        """对密码进行哈希处理"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register(self, username, password, role='普通用户'):
        """注册新用户"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查用户名是否已存在
            cursor.execute('SELECT username FROM users WHERE username = ?', (username,))
            if cursor.fetchone() is not None:
                return False, '用户名已存在'
            
            # 验证角色是否有效
            valid_roles = ['普通用户', '医生', '管理员']
            if role not in valid_roles:
                return False, '无效的用户角色'
            
            # 插入新用户
            hashed_password = self._hash_password(password)
            cursor.execute(
                'INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
                (username, hashed_password, role)
            )
            
            conn.commit()
            return True, '注册成功'
        except Exception as e:
            return False, f'注册失败: {str(e)}'
        finally:
            conn.close()
    
    def login(self, username, password):
        """用户登录"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 验证用户名和密码
            hashed_password = self._hash_password(password)
            cursor.execute(
                'SELECT id, username, role FROM users WHERE username = ? AND password = ?',
                (username, hashed_password)
            )
            user = cursor.fetchone()
            
            if user is None:
                return False, '用户名或密码错误', None
            
            # 更新最后登录时间
            cursor.execute(
                'UPDATE users SET last_login = ? WHERE id = ?',
                (datetime.now(), user[0])
            )
            
            conn.commit()
            return True, '登录成功', {'id': user[0], 'username': user[1], 'role': user[2]}
        except Exception as e:
            return False, f'登录失败: {str(e)}', None
        finally:
            conn.close()
    
    def get_user_info(self, user_id):
        """获取用户信息"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT username, role, created_at, last_login FROM users WHERE id = ?',
                (user_id,)
            )
            user = cursor.fetchone()
            
            if user is None:
                return None
            
            return {
                'username': user[0],
                'role': user[1],
                'created_at': user[2],
                'last_login': user[3]
            }
        finally:
            conn.close()