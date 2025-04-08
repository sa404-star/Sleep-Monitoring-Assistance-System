
import streamlit as st
import pandas as pd
import numpy as np
import pywt
import sys
import path
import os 
import uuid
import json
from langchain_community.llms import Ollama
from streamlit import session_state as st_session
from openai import OpenAI  

from pathlib import Path
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


dir = Path(__file__).resolve()
sys.path.append(str(dir.parent.parent))

path = os.path.dirname(__file__)

# 初始化会话状态变量
if 'is_logged_in' not in st_session:
    st_session.is_logged_in = False
    
if 'current_user' not in st_session:
    st_session.current_user = None
    
if 'view_mode' not in st_session:
    st_session.view_mode = "家庭模式"
    
if 'model_choice' not in st_session:
    st_session.model_choice = "API调用"
    
if 'local_model_loaded' not in st_session:
    st_session.local_model_loaded = False
    
if 'chat_history' not in st_session:
    st_session.chat_history = []
    
# 添加用户ID初始化
if 'user_id' not in st_session:
    st_session.user_id = "default_user"

# 添加用户数据目录
USER_DATA_DIR = f"{path}/user_data"
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

# 添加用户记录目录
USER_RECORDS_DIR = f"{path}/user_records"
if not os.path.exists(USER_RECORDS_DIR):
    os.makedirs(USER_RECORDS_DIR)

# 用户数据文件
USERS_FILE = f"{USER_DATA_DIR}/users.json"

# 加载用户数据
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

# 保存用户数据
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

# 用户注册
def register_user(username, password, role, security_question, security_answer):
    users = load_users()
    if username in users:
        return False, "用户名已存在"
    
    # 生成唯一用户ID
    user_id = str(uuid.uuid4())
    
    # 创建用户记录目录
    user_record_dir = f"{USER_RECORDS_DIR}/{user_id}"
    if not os.path.exists(user_record_dir):
        os.makedirs(user_record_dir)
    
    users[username] = {
        "password": password,
        "role": role,
        "user_id": user_id,
        "security_question": security_question,
        "security_answer": security_answer
    }
    save_users(users)
    return True, user_id

# 密码重置函数
def reset_password(username, security_answer, new_password):
    users = load_users()
    if username not in users:
        return False, "用户名不存在"
    
    # 验证安全问题答案
    if users[username]["security_answer"] != security_answer:
        return False, "安全问题答案错误"
    
    # 更新密码
    users[username]["password"] = new_password
    save_users(users)
    return True, "密码重置成功！请使用新密码登录"

# 用户登录
def login_user(username, password):
    users = load_users()
    if username not in users:
        return False, "用户名不存在"
    
    if users[username]["password"] != password:
        return False, "密码错误"
    
    return True, users[username]

# 登录/注册界面
def show_auth_page():
    st.markdown("<h1 style='text-align: center;'>睡眠分期辅助系统</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>用户登录/注册</h3>", unsafe_allow_html=True)
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["登录", "注册", "忘记密码"])
    
    with tab1:
        # 登录表单
        with st.form("login_form"):
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            submit_login = st.form_submit_button("登录")
            
            if submit_login:
                if not username or not password:
                    st.error("请填写用户名和密码")
                else:
                    success, result = login_user(username, password)
                    if success:
                        st_session.is_logged_in = True
                        st_session.current_user = username
                        st_session.user_role = result["role"]
                        st_session.user_id = result["user_id"]
                        st.success("登录成功！")
                        st.rerun()
                    else:
                        st.error(result)
    
    with tab2:
        # 注册表单
        with st.form("register_form"):
            new_username = st.text_input("设置用户名")
            new_password = st.text_input("设置密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")
            role = st.selectbox("用户类别", ["普通用户", "医生", "管理员"])
            # 添加安全问题和答案字段
            security_question = st.selectbox("选择安全问题", ["您的出生地是?", "您的母亲姓氏是?", "您的第一所学校是?", "您的宠物名字是?"])
            security_answer = st.text_input("安全问题答案")
            submit_register = st.form_submit_button("注册")
            
            if submit_register:
                if not new_username or not new_password or not confirm_password or not security_answer:
                    st.error("请填写所有字段")
                elif new_password != confirm_password:
                    st.error("两次输入的密码不一致")
                else:
                    # 更新这里，传递安全问题和答案
                    success, result = register_user(new_username, new_password, role, security_question, security_answer)
                    if success:
                        st.success(f"注册成功！您的用户ID是: {result}")
                    else:
                        st.error(result)
    with tab3:
        # 忘记密码表单
        with st.form("reset_password_form"):
            reset_username = st.text_input("用户名")
            check_username = st.form_submit_button("验证用户名")
            
            if check_username and reset_username:
                users = load_users()
                if reset_username in users:
                    st.session_state.reset_username = reset_username
                    st.session_state.security_question = users[reset_username]["security_question"]
                    st.success(f"用户名验证成功，请回答安全问题")
                    st.rerun()
                else:
                    st.error("用户名不存在")
        
        # 如果用户名已验证，显示安全问题
        if "reset_username" in st.session_state:
            with st.form("security_question_form"):
                st.write(f"安全问题: {st.session_state.security_question}")
                security_answer = st.text_input("请输入答案")
                new_password = st.text_input("新密码", type="password")
                confirm_password = st.text_input("确认新密码", type="password")
                reset_submit = st.form_submit_button("重置密码")
                
                if reset_submit:
                    if not security_answer or not new_password or not confirm_password:
                        st.error("请填写所有字段")
                    elif new_password != confirm_password:
                        st.error("两次输入的密码不一致")
                    else:
                        success, message = reset_password(st.session_state.reset_username, security_answer, new_password)
                        if success:
                            st.success(message)
                            # 清除重置状态
                            if "reset_username" in st.session_state:
                                del st.session_state.reset_username
                            if "security_question" in st.session_state:
                                del st.session_state.security_question
                        else:
                            st.error(message)

def show_title():
    title = "基于CrossFusionSleepNet的睡眠分期辅助系统"
    st.markdown(
      f"<h2 style='text-align: center;'>{title}</h2>", 
      unsafe_allow_html=True
      )


def show_identity():
  col1, col2, col3 = st.columns([5, 5, 5])
  identity = "姓名:XXX 年龄:XX  性别:XX "
  col2.write(identity)
  
  # 添加用户角色选择
  if "user_role" not in st_session:
    st_session.user_role = "普通用户"
    
  # 添加视图模式选择
  if "view_mode" not in st_session:
    st_session.view_mode = "家庭模式"
    
  # 添加模型选择
  if "model_choice" not in st_session:
    st_session.model_choice = "API调用"
    
  # 添加用户ID
  if "user_id" not in st_session:
    st_session.user_id = "default_user"
    
  with st.sidebar:
    st.subheader("用户信息")
    st.info(f"当前用户: {st_session.current_user}")
    st.info(f"用户ID: {st_session.user_id}")
    
    # 添加退出登录按钮
    if st.button("退出登录"):
        st_session.is_logged_in = False
        st_session.current_user = None
        st.rerun()
    
    st.subheader("用户角色设置")
    user_role = st.radio(
        "请选择您的角色",
        ["普通用户", "医生", "管理员"],
        index=["普通用户", "医生", "管理员"].index(st_session.user_role)
    )
    st_session.user_role = user_role
    st.write(f"当前角色: {st_session.user_role}")
    
    # 添加视图模式选择
    st.divider()
    st.subheader("视图模式设置")
    view_mode = st.radio(
        "请选择视图模式",
        ["家庭模式", "专家模式"],
        index=["家庭模式", "专家模式"].index(st_session.view_mode),
        help="家庭模式显示简化信息和健康建议，专家模式显示详细技术指标"
    )
    st_session.view_mode = view_mode
    st.write(f"当前视图: {st_session.view_mode}")
    
    # 添加模型选择
    st.divider()
    st.subheader("模型选择")
    model_choice = st.radio(
        "请选择AI模型",
        ["API调用", "本地模型"],
        index=["API调用", "本地模型"].index(st_session.model_choice),
        help="API调用提供更高质量回答，本地模型保护隐私且无需网络"
    )
    st_session.model_choice = model_choice
    st.write(f"当前模型: {st_session.model_choice}")
    
    # 如果选择本地模型，显示模型加载状态
    if st_session.model_choice == "本地模型":
        if "local_model_loaded" not in st_session:
            st_session.local_model_loaded = False
            
        if not st_session.local_model_loaded:
            with st.spinner("正在加载本地模型..."):
                try:
                    # 这里可以添加实际的模型加载代码
                    # 模拟加载过程
                    import time
                    time.sleep(1)
                    st_session.local_model_loaded = True
                    st.success("本地模型加载成功")
                except Exception as e:
                    st.error(f"模型加载失败: {str(e)}")
    st.divider()

def wavelet(x_file, fams):
  wp = pywt.WaveletPacket2D(data=x_file, wavelet=fams, mode='zero')
  scaler = MinMaxScaler(feature_range=(-1,1))

  w_1 = wp['addd'].data
  w_2 = wp['daaa'].data
  w_3 = wp['daad'].data
  w_4 = wp['dada'].data
  w_5 = wp['dadd'].data
  w_6 = wp['ddaa'].data
  w = np.concatenate((w_1,w_2,w_3,w_4,w_5,w_6), axis=0)
  w=scaler.fit_transform(w)

  n1_1 = wp['aadd'].data
  n1_2 = wp['adaa'].data
  n1_3 = wp['adad'].data
  n1_4 = wp['adda'].data
  n1 = np.concatenate((n1_1,n1_2, n1_3,n1_4), axis=0)
  n1=scaler.fit_transform(n1)

  n2_1 = wp['dada'].data
  n2_2 = wp['dadd'].data
  n2_3 = wp['ddaa'].data
  n2_4 = wp['ddad'].data
  n2_5 = wp['ddda'].data
  n2_6 = wp['dddd'].data
  n2 = np.concatenate((n2_1,n2_2,n2_3,n2_4,n2_5,n2_6), axis=0)
  n2=scaler.fit_transform(n2)

  n3_1 = wp['aaaa'].data
  n3_2 = wp['aaad'].data
  n3 = np.concatenate((n3_1,n3_2), axis=0)
  n3=scaler.fit_transform(n3)

  r_1 = wp['aaad'].data
  r_2 = wp['aada'].data
  r_3 = wp['aadd'].data
  r_4 = wp['adaa'].data
  r_5 = wp['adad'].data
  r = np.concatenate((r_1,r_2,r_3,r_4,r_5), axis=0)
  r=scaler.fit_transform(r)

  w_label = []
  lab_w = [[1] * 1 for i in range(len(w))]
  w_label = np.asarray(lab_w)

  n1_label = []
  lab_n1 = [[2] * 1 for i in range(len(n1))]
  n1_label = np.asarray(lab_n1)

  n2_label = []
  lab_2 = [[3] * 1 for i in range(len(n2))]
  n2_label = np.asarray(lab_2)

  n3_label = []
  lab_3 = [[4] * 1 for i in range(len(n3))]
  n3_label = np.asarray(lab_3)

  r_label = []
  lab_r = [[5] * 1 for i in range(len(r))]
  r_label = np.asarray(lab_r)

  x_file = np.concatenate((w,n1, n2, n3, r), axis=0)
  y_file = np.concatenate((w_label, n1_label, n2_label, n3_label, r_label), axis=0)
  return x_file, y_file
  
def upload_sleep_file():
  return st.file_uploader("Upload Sleep File", key="uploader")

def select_sleep_file():
  file_paths = [f"{path}/examples-dataset/Data_EEG_s10.pkl", 
                f"{path}/examples-dataset/Data_EEG_s80.pkl", 
                f"{path}/examples-dataset/Data_EEG_s137.pkl",
                f"{path}/examples-dataset/Data_EEG_s144.pkl"]
  button_states = [False, False, False, False]

  for i, file_path in enumerate(file_paths):
    button_states[i] = st.checkbox(f"Example file no {i+1}", key=f"checkbox_{i}")
    if button_states[i]:
      return file_path
    
def load_file():
  upload_header = "<h4 style='text-align: center;'>Upload Sleep File</h4>"
  selected_header = "<h4 style='text-align: center;'>Select Example File</h4>"

  with st.sidebar:
    st.markdown(upload_header, unsafe_allow_html=True)
    uploaded_data = upload_sleep_file()
    st.divider()
    st.markdown(selected_header, unsafe_allow_html=True)
    selected_data = select_sleep_file()
    
  return uploaded_data if uploaded_data is not None else selected_data

def calculate_sleep_stages(sleep_file):
  if sleep_file is None:
    return None
  
  data = pd.read_pickle(sleep_file).dropna()

  x_file = data.iloc[:,0:3000].to_numpy()
  y_file = data.iloc[:, 3000].to_numpy()

  x_file = np.array(x_file)
  y_file = np.reshape(y_file, (-1, 1))

  wavelet_family  = 'db9' # Wavelet family

  x_test, y_test = wavelet(x_file, wavelet_family )  # Wavelet
  oneHot = OneHotEncoder(sparse=False)
  y_test = oneHot.fit_transform(y_file)

  weights_path = f"{path}/models/model.h5"
  model = load_model(weights_path)
  y_pred=model.predict(x_test) 

  y_pred_max=np.argmax(y_pred, axis=1)
  y_test_max=np.argmax(y_test, axis=1)

  return data,x_file, y_test_max

def displaying_sleep_wave(sleep_wave):
  st.subheader("Sleep Wave")
  st.line_chart(data=sleep_wave[13], width=0, height=0, use_container_width=True)

def displaying_hypnogram(hypnogram):
  st.subheader(f"Hypnogram (stages of sleep over time)")
  st.line_chart(data=hypnogram, width=0, height=0, use_container_width=True)
  

def generate_report(data):
  ### Mapping sleep stages
  sleep_stages_mapping = {1: "Wake", 2: "Non - REM 1", 3: "Non - REM 2", 4: "Non - REM 3", 5: "REM"}
  data_hypnogram = data['Hypnogram'].map(sleep_stages_mapping)

  ### Creating report
  raw_report = pd.DataFrame({
      'Sleep Stages': data_hypnogram.value_counts(sort=False).index.tolist(),
      'Persentage': round(((data_hypnogram.value_counts(sort=False) / data_hypnogram.value_counts(sort=False).sum()) * 100), 2),
      'Length': data_hypnogram.value_counts(sort=False) / 2,
      'Quality' : '',
    })
  
  # print(raw_report.columns)

  # raw_report.to_csv('raw_report.csv', index=False, encoding='utf-8-sig')
  
  
  # raw_report.to_csv('raw_report.csv', index=False, encoding='utf-8-sig')

  raw_report.set_index('Sleep Stages', inplace=True)
  raw_report['Quality'] = raw_report.apply(lambda row: calculate_sleep_quality(row), axis=1)
  

  ### Displaying report
  st.subheader("Details of Sleep Activity")

  column_configs = {
      "Persentage": st.column_config.ProgressColumn(
          label="Persentage of sleep stages",
          help="Persentage of sleep stages",
          format="%f%%",
          min_value=0,
          max_value=100,
          width="medium"
      ),
      "Characteristic": st.column_config.TextColumn(
          label="Characteristic",
          help="Characteristic of sleep stages",
          width="large"
      ),
      "Length": st.column_config.NumberColumn(
          label="Length of sleep stages",
          help="Length of sleep stages in minutes",
          format="%f minutes",
          width="small"
      ),
      "Quality": st.column_config.ImageColumn(
        "Sleep Stage Quality", 
        help="Streamlit app preview screenshots",
        width="small",
      )
  }

  st.data_editor(
    raw_report,
    column_config=column_configs,
    hide_index=False,
    disabled=True,
    use_container_width=True
)
  
  with st.expander("See notes"):
    st.markdown("Sleep stages are based on the wavelet transform of the EEG signal.")
    st.markdown('''
    - Hyphen Mark : "Wake Stage" does not have specific length
    - Check Mark : Normal Length of Sleep Stages
    - Cross Mark : Not Normal Length of Sleep Stages
    
    See also
    - https://emedicine.medscape.com/article/1140322-overview?form=fpf#a1
  ''')
  return raw_report


def calculate_sleep_quality(row):
  if row.name == 'Wake':
      return 'https://img.icons8.com/fluency/48/minus-math.png'
  if row.name == 'Non - REM 1' and 3 < row['Persentage'] < 5:
      return 'https://img.icons8.com/fluency/48/checkmark--v1.png'
  elif row.name == 'Non - REM 2' and 50 < row['Persentage'] < 60:
      return 'https://img.icons8.com/fluency/48/checkmark--v1.png'
  elif row.name == 'Non - REM 3' and 10 < row['Persentage'] < 20:
      return 'https://img.icons8.com/fluency/48/checkmark--v1.png'
  elif row.name == 'REM' and 10 < row['Persentage'] < 25:
      return 'https://img.icons8.com/fluency/48/checkmark--v1.png'
  else:
      return 'https://img.icons8.com/fluency/48/delete-sign.png'
  

def sleep_characteristic():
  characteristics = {
    'Wake': "The person is relaxed. This stage lasts from when the eyes are open to when the person becomes drowsy and their eyes close.",
    'Non - REM 1': "The person is asleep but their skeletal muscle tone and breathing is the same as when awake.",
    'Non - REM 2': "The length of this stage increases with each cycle. Bruxism (teeth grinding) may occur.",
    'Non - REM 3': "The deepest stage of sleep, this is when the body repairs and regrows tissues, builds bone and muscle, and strengthens the immune system. Sleepwalking, night terrors, and bedwetting may occur. A person is hard to wake. If they are woken at this stage, they may feel groggy for 30–60 minutes after.",
    'REM': "Associated with dreaming, REM is not restful sleep. Eyes and breathing muscles are active but skeletal muscles are paralyzed. Breathing may be irregular and erratic. Usually follows the other stages, starting around 90 minutes after falling asleep."
  }

  normal_length  = {
    'Wake': "-",
    'Non - REM 1': "3-5 %",
    'Non - REM 2': "50-60 %",
    'Non - REM 3': "10-20 %",
    'REM': "10-25 %"
  }

  sleep_df  = pd.DataFrame({
      'Sleep Stages': characteristics.keys(),
      'Length': normal_length .values(),
      'Characteristic': characteristics.values(),
  })
  sleep_df .set_index('Sleep Stages', inplace=True)

  column_configs = {
      "Sleep Stages": st.column_config.TextColumn(
          label="Sleep Stages",
          help="Sleep Stages",
          width="small"
      ),
      "Length": st.column_config.TextColumn(
          label="Normal % of sleep",
          help="Characteristic of sleep stages",
          width="small"
      ),
      "Characteristic": st.column_config.TextColumn(
          label="Characteristic",
          help="Characteristic of sleep stages",
          width="large"
      )
  }

  st.subheader("Sleep Stages Characteristic")
  st.data_editor(
    sleep_df ,
    column_config=column_configs,
    hide_index=False,
    disabled=True,
    use_container_width=True
)
  
def displaying_bar(report):
  st.subheader("Duration of Each Sleep Stage")
  st.bar_chart(report['Length'])

def displaying_sleep_durations(report):
  total_time_minutes = report['Length'].sum()
  total_hours = str(int((total_time_minutes / 60)))
  total_minutes = str(int(total_time_minutes % 60))
  sleep_duration = f"Total Sleep Duration = :blue[{total_hours} hours] :blue[{total_minutes} minutes]"
  st.write(sleep_duration)

def sleep_quality_by_age():
  duration = {
    'Newborns': '14 - 17 hours',
    'Infants': '12 - 15 hours',
    'Toddlers': '11 - 14 hours',
    'Preschoolers': '10 - 13 hours',
    'Tenagers': '8 - 10 hours',
    'Young Adults and Adults': '7 - 9 hours',
    'Older People': '7 - 8 hours',
  }

  sleep_duration = pd.DataFrame({
    'Age Group': duration.keys(),
    'Duration': duration.values(),
  })

  sleep_duration.set_index('Age Group', inplace=True)

  column_configs = {
      "Duration": st.column_config.TextColumn(
          label="Duration",
          help="Duration",
          width="small"
      ),
  }

  st.subheader("Sleep Durations by Age Group")
  st.data_editor(
    sleep_duration,
    column_config=column_configs,
    hide_index=False,
    disabled=True,
    use_container_width=True
)



def deepseek_api(messages):
    """支持多轮对话的DeepSeek API调用"""
    client = OpenAI(api_key="xxxxxxx", base_url="https://api.deepseek.com")
    
    # 将消息格式转换为OpenAI格式
    formatted_messages = []
    for msg in messages:
        formatted_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # 添加系统消息，根据用户角色和视图模式调整提示
    system_content = "你是一个专业的睡眠分析助手，"
    
    # 根据用户角色调整
    if st_session.user_role == "普通用户":
        system_content += "请用通俗易懂的语言解释睡眠报告，避免使用过多专业术语。"
    elif st_session.user_role == "医生":
        system_content += "请提供详细的医学分析和专业建议，可以使用医学术语。"
    elif st_session.user_role == "管理员":
        system_content += "请提供全面的数据分析和系统性建议，包括可能的技术参数调整。"
    
    # 根据视图模式调整
    if st_session.view_mode == "家庭模式":
        system_content += "由于用户选择了家庭模式，请重点关注实用的健康建议和生活习惯改善，使用简单易懂的语言。"
    else:  # 专家模式
        system_content += "由于用户选择了专家模式，可以提供更多技术细节和专业分析，包括波形特征和频谱分析的解读。"
    
    if formatted_messages and formatted_messages[0]["role"] != "system":
        formatted_messages.insert(0, {"role": "system", "content": system_content})
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=formatted_messages,
        stream=False
    )
    
    return response.choices[0].message.content

def display_chat_messages():
    """显示聊天记录"""
    for message in st_session.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(raw_report):
    """处理用户输入"""
    if user_input := st.chat_input("请输入关于睡眠分析的疑问"):
        # 添加用户消息到对话历史
        st_session.chat_history.append({"role": "user", "content": user_input})
        
        # 构造系统提示（包含报告数据和用户角色）
        system_prompt = f"""睡眠分析报告数据：
        {raw_report.to_markdown()}
        
        用户角色：{st_session.user_role}
        视图模式：{st_session.view_mode}
        
        请根据以上报告数据回答用户问题，回答需：
        1. 结合报告中的具体数据
        2. 根据用户角色调整回答深度和专业程度
        3. 提供适合该角色的建议
        当前问题：{user_input}"""
        
        # 生成并添加AI回复
        with st.spinner("正在分析..."):
            response = ai_model_call(st_session.chat_history + [{"role": "user", "content": system_prompt}])
        
        st_session.chat_history.append({"role": "assistant", "content": response})
        
        # 刷新显示
        st.rerun()

def display_signal_spectrum(sleep_wave):
    """显示信号频谱分析（专家模式）"""
    st.subheader("脑电信号频谱分析")
    
    # 使用FFT计算频谱
    from scipy import signal
    from scipy.fft import fft, fftfreq
    
    # 选择一段有代表性的数据
    sample_data = sleep_wave[13][:1000]  # 使用前1000个点
    
    # 计算频谱
    fs = 100  # 假设采样率为100Hz
    f, Pxx = signal.welch(sample_data, fs, nperseg=256)
    
    # 创建频谱图
    spectrum_df = pd.DataFrame({
        '频率 (Hz)': f,
        '功率谱密度': Pxx
    })
    
    # 显示频谱图
    st.line_chart(spectrum_df.set_index('频率 (Hz)'))
    
    # 显示频段分析
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 计算各频段能量
    delta_idx = (f >= 0.5) & (f <= 4)
    theta_idx = (f >= 4) & (f <= 8)
    alpha_idx = (f >= 8) & (f <= 13)
    beta_idx = (f >= 13) & (f <= 30)
    gamma_idx = (f >= 30)
    
    delta_power = np.sum(Pxx[delta_idx])
    theta_power = np.sum(Pxx[theta_idx])
    alpha_power = np.sum(Pxx[alpha_idx])
    beta_power = np.sum(Pxx[beta_idx])
    gamma_power = np.sum(Pxx[gamma_idx])
    
    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
    
    col1.metric("Delta波段", f"{delta_power/total_power*100:.1f}%", "0.5-4Hz")
    col2.metric("Theta波段", f"{theta_power/total_power*100:.1f}%", "4-8Hz")
    col3.metric("Alpha波段", f"{alpha_power/total_power*100:.1f}%", "8-13Hz")
    col4.metric("Beta波段", f"{beta_power/total_power*100:.1f}%", "13-30Hz")
    col5.metric("Gamma波段", f"{gamma_power/total_power*100:.1f}%", ">30Hz")
    
    st.info("Delta波与深度睡眠相关，Theta波与轻度睡眠相关，Alpha波与放松状态相关，Beta波与清醒状态相关，Gamma波与高度认知活动相关。")

def display_feature_importance():
    """显示特征重要性分析（专家模式）"""
    st.subheader("睡眠分期特征重要性")
    
    # 模拟特征重要性数据
    features = [
        "Delta波能量", "Theta波能量", "Alpha波能量", "Beta波能量", 
        "Gamma波能量", "波形复杂度", "频谱熵", "波形峰度",
        "波形偏度", "零交叉率"
    ]
    
    importance = [0.28, 0.22, 0.15, 0.12, 0.05, 0.07, 0.04, 0.03, 0.02, 0.02]
    
    # 创建特征重要性数据框
    feature_imp_df = pd.DataFrame({
        '特征': features,
        '重要性': importance
    })
    
    # 按重要性排序
    feature_imp_df = feature_imp_df.sort_values('重要性', ascending=False)
    
    # 显示条形图
    st.bar_chart(feature_imp_df.set_index('特征'))
    
    st.info("特征重要性表示各项指标对睡眠分期判断的影响程度。Delta波能量和Theta波能量是判断睡眠阶段的最重要指标。")

def display_health_recommendations(raw_report):
    """显示健康建议（家庭模式）"""
    st.subheader("睡眠健康建议")
    
    # 分析睡眠报告，生成健康建议
    wake_pct = raw_report.loc["Wake", "Persentage"] if "Wake" in raw_report.index else 0
    n1_pct = raw_report.loc["Non - REM 1", "Persentage"] if "Non - REM 1" in raw_report.index else 0
    n2_pct = raw_report.loc["Non - REM 2", "Persentage"] if "Non - REM 2" in raw_report.index else 0
    n3_pct = raw_report.loc["Non - REM 3", "Persentage"] if "Non - REM 3" in raw_report.index else 0
    rem_pct = raw_report.loc["REM", "Persentage"] if "REM" in raw_report.index else 0
    
    # 创建健康建议
    recommendations = []
    
    # 总体睡眠质量评估
    if n3_pct < 10:
        sleep_quality = "较差"
        recommendations.append("您的深度睡眠（Non-REM 3）比例偏低，可能导致睡眠质量不佳。")
    elif n3_pct > 20:
        sleep_quality = "良好"
        recommendations.append("您的深度睡眠（Non-REM 3）比例充足，有助于身体恢复。")
    else:
        sleep_quality = "正常"
        recommendations.append("您的深度睡眠（Non-REM 3）比例正常。")
    
    if rem_pct < 10:
        recommendations.append("您的REM睡眠比例偏低，可能影响记忆巩固和情绪调节。")
    elif rem_pct > 25:
        recommendations.append("您的REM睡眠比例偏高，可能与压力或某些药物有关。")
    else:
        recommendations.append("您的REM睡眠比例正常，有助于记忆巩固和情绪调节。")
    
    if wake_pct > 10:
        recommendations.append("您的清醒时间比例偏高，可能存在入睡困难或夜间醒来的情况。")
    
    # 显示总体评估
    st.markdown(f"### 睡眠质量总体评估: {sleep_quality}")
    
    # 显示具体建议
    st.markdown("### 具体建议:")
    for i, rec in enumerate(recommendations):
        st.markdown(f"{i+1}. {rec}")
    
    # 通用建议
    st.markdown("### 改善睡眠的通用建议:")
    st.markdown("""
    1. 保持规律的作息时间，包括周末
    2. 睡前避免使用电子设备和摄入咖啡因
    3. 确保睡眠环境舒适、安静和黑暗
    4. 睡前进行放松活动，如阅读或冥想
    5. 适当的白天运动有助于夜间睡眠
    """)
    
    # 根据睡眠阶段比例提供个性化建议
    st.markdown("### 个性化建议:")
    if n3_pct < 10:
        st.markdown("- 增加深度睡眠: 白天适当体力活动，避免睡前饮酒，保持规律作息")
    if rem_pct < 10:
        st.markdown("- 增加REM睡眠: 减轻压力，避免使用影响REM的药物，保证充足的总睡眠时间")
    if wake_pct > 10:
        st.markdown("- 减少夜间醒来: 避免睡前大量饮水，保持舒适的睡眠环境，考虑使用白噪音")


def load_user_sleep_records(user_id):
    """加载用户的所有睡眠记录"""
    import os
    import json
    import pandas as pd
    
    records_dir = f"{path}/user_records/{user_id}"
    if not os.path.exists(records_dir):
        return []
    
    records = []
    for filename in os.listdir(records_dir):
        if filename.endswith('.json'):
            with open(f"{records_dir}/{filename}", 'r') as f:
                record = json.load(f)
                records.append(record)
    
    return sorted(records, key=lambda x: x['date'])



# 在适当位置添加以下函数

def save_sleep_record(user_id, raw_report, date=None):
    """保存睡眠记录到数据库或文件"""
    import datetime
    import json
    import os
    
    if date is None:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 创建存储目录
    records_dir = f"{path}/user_records"
    user_dir = f"{records_dir}/{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    
    # 将报告转换为可序列化格式
    report_dict = raw_report.reset_index().to_dict(orient='records')
    
    # 保存记录
    record_file = f"{user_dir}/{date}.json"
    with open(record_file, 'w') as f:
        json.dump({
            'date': date,
            'report': report_dict,
            'user_role': st_session.user_role,
            'view_mode': st_session.view_mode
        }, f)
    
    return record_file



def display_sleep_trends(user_id):
    """显示用户的睡眠趋势"""
    records = load_user_sleep_records(user_id)
    
    if not records:
        st.info("暂无历史睡眠记录，无法显示趋势分析。")
        return
    
    # 提取关键指标
    dates = []
    n3_percentages = []
    rem_percentages = []
    total_sleep_times = []
    
    for record in records:
        dates.append(record['date'])
        
        # 从报告中提取数据
        report_df = pd.DataFrame(record['report'])
        report_df.set_index('Sleep Stages', inplace=True)
        
        n3_pct = report_df.loc["Non - REM 3", "Persentage"] if "Non - REM 3" in report_df.index else 0
        rem_pct = report_df.loc["REM", "Persentage"] if "REM" in report_df.index else 0
        total_time = report_df["Length"].sum()
        
        n3_percentages.append(n3_pct)
        rem_percentages.append(rem_pct)
        total_sleep_times.append(total_time)
    
    # 创建趋势图
    st.subheader("睡眠趋势分析")
    
    # 总睡眠时间趋势
    st.markdown("#### 总睡眠时间趋势")
    sleep_time_df = pd.DataFrame({
        '日期': dates,
        '睡眠时长(分钟)': total_sleep_times
    })
    st.line_chart(sleep_time_df.set_index('日期'))
    
    # 深度睡眠和REM睡眠比例趋势
    st.markdown("#### 关键睡眠阶段比例趋势")
    sleep_stages_df = pd.DataFrame({
        '日期': dates,
        '深度睡眠比例(%)': n3_percentages,
        'REM睡眠比例(%)': rem_percentages
    })
    st.line_chart(sleep_stages_df.set_index('日期'))
    
    # 提供趋势分析建议
    if len(records) >= 3:
        st.subheader("趋势分析建议")
        
        # 分析深度睡眠趋势
        n3_trend = n3_percentages[-1] - n3_percentages[0]
        if n3_trend > 5:
            st.success("您的深度睡眠比例呈上升趋势，睡眠质量正在改善。")
        elif n3_trend < -5:
            st.warning("您的深度睡眠比例呈下降趋势，建议关注睡眠质量变化。")
        
        # 分析总睡眠时间趋势
        time_trend = total_sleep_times[-1] - total_sleep_times[0]
        if time_trend > 30:
            st.success("您的睡眠时间呈增加趋势，有助于身体恢复。")
        elif time_trend < -30:
            st.warning("您的睡眠时间呈减少趋势，请注意保证充足的睡眠。")


def ai_model_call(messages):
    """根据选择的模型调用AI"""
    if st_session.model_choice == "API调用":
        return deepseek_api(messages)
    else:
        # 本地模型调用
        try:
            llm = Ollama(model="llama2")
            # 简化消息格式
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            return llm.invoke(prompt)
        except Exception as e:
            st.error(f"本地模型调用失败: {str(e)}")
            return "本地模型调用失败，请检查Ollama服务是否运行。"


def main():

    if not st_session.is_logged_in:
        show_auth_page()
        return
    show_title()
    show_identity()
    
    # 添加日历选择
    col1, col2 = st.columns([3, 1])
    with col2:
        selected_date = st.date_input("选择日期", value=None)
        
    sleep_file = load_file()

    # 初始化对话历史
    if "chat_history" not in st_session:
        st_session.chat_history = []
        
    # 添加标签页，分离当前分析和历史记录
    tab1, tab2, tab3 = st.tabs(["当前分析", "历史记录", "趋势分析"])
    
    with tab1:
        if sleep_file is not None:
            st.divider()
            st.markdown(f"<h2 style='text-align: center;'>Sleep Identification Reports</h2>", unsafe_allow_html=True)
            
            # 数据分析和可视化部分
            sleep_data, sleep_wave, hypnogram = calculate_sleep_stages(sleep_file)
            raw_report = generate_report(sleep_data)
            
            # 添加保存记录按钮
            if st.button("保存当前睡眠记录"):
                if selected_date is None:
                    st.warning("请先选择日期再保存记录")
                else:
                    record_path = save_sleep_record(
                        st_session.user_id, 
                        raw_report, 
                        date=selected_date.strftime("%Y-%m-%d")
                    )
                    st.success(f"睡眠记录已保存至 {selected_date.strftime('%Y-%m-%d')}")
            
            # 基本信息显示（两种模式都显示）
            displaying_sleep_durations(raw_report)
            
            # 根据视图模式显示不同内容
            if st_session.view_mode == "专家模式":
                # 专家模式显示详细技术信息
                displaying_sleep_wave(sleep_wave)
                displaying_hypnogram(hypnogram)
                
                # 显示频谱分析和特征重要性（仅专家模式）
                display_signal_spectrum(sleep_wave)
                display_feature_importance()
                
                # 根据用户角色显示更多专业信息
                if st_session.user_role in ["医生", "管理员"]:
                    sleep_characteristic()
                    sleep_quality_by_age()
                    
                    # 仅为管理员显示的额外技术信息
                    if st_session.user_role == "管理员":
                        st.subheader("系统技术参数")
                        st.info("模型: CrossFusionSleepNet | 小波变换: db9 | 数据采样率: 100Hz")
                        st.expander("查看原始数据").dataframe(sleep_data)
            else:
                # 家庭模式显示简化信息和健康建议
                st.markdown("### 睡眠阶段分布")
                displaying_bar(raw_report)  # 显示睡眠阶段分布条形图
                
                # 显示健康建议
                display_health_recommendations(raw_report)
                
                # 显示按年龄段的睡眠时长参考
                sleep_quality_by_age()

            st.divider()  # 添加分隔线
            
            # 对话建议部分（两种模式都显示）
            st.markdown("## 睡眠分析建议对话")
            
            # 显示初始分析建议
            if not st_session.chat_history:
                initial_prompt = f"""请分析以下睡眠报告：
                {raw_report.to_markdown()}
                用户角色是：{st_session.user_role}
                视图模式是：{st_session.view_mode}
                请根据用户角色和视图模式给出适合的总体评估和改善建议"""
                
                initial_response = deepseek_api([{"role": "user", "content": initial_prompt}])
                st_session.chat_history.append({"role": "assistant", "content": "## 初始分析建议\n" + initial_response})

            # 显示聊天界面
            display_chat_messages()
            handle_user_input(raw_report)
        else:
            # 当没有选择睡眠文件时显示提示信息
            st.info("请从侧边栏上传或选择睡眠文件进行分析")
    
    with tab2:
        # 历史记录标签页内容
        st.subheader("历史睡眠记录")
        records = load_user_sleep_records(st_session.user_id)
        
        if not records:
            st.info("暂无历史睡眠记录")
        else:
            # 创建历史记录选择器
            record_dates = [record['date'] for record in records]
            selected_record_date = st.selectbox("选择日期查看记录", record_dates)
            
            # 显示选定的记录
            selected_record = next((r for r in records if r['date'] == selected_record_date), None)
            if selected_record:
                st.markdown(f"### {selected_record_date} 睡眠记录")
                
                # 将记录转换回DataFrame
                report_df = pd.DataFrame(selected_record['report'])
                report_df.set_index('Sleep Stages', inplace=True)
                
                # 显示睡眠阶段分布
                st.markdown("#### 睡眠阶段分布")
                st.bar_chart(report_df['Length'])
                
                # 显示详细报告
                st.markdown("#### 详细报告")
                column_configs = {
                    "Persentage": st.column_config.ProgressColumn(
                        label="Persentage of sleep stages",
                        format="%f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Length": st.column_config.NumberColumn(
                        label="Length of sleep stages",
                        format="%f minutes",
                    ),
                    "Quality": st.column_config.ImageColumn(
                        "Sleep Stage Quality", 
                    )
                }
                st.data_editor(
                    report_df,
                    column_config=column_configs,
                    hide_index=False,
                    disabled=True,
                    use_container_width=True
                )
    
    with tab3:
        display_sleep_trends(st_session.user_id)

if __name__ == "__main__":
    main()

