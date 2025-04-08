import streamlit as st
import pandas as pd
import numpy as np
import pywt
import sys
import path
import os 
from langchain_community.llms import Ollama
from streamlit import session_state as st_session

from pathlib import Path
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import time
from langchain_community.llms import Ollama
import statistics

# dir = path.Path(__file__).abspath()
# sys.path.append(dir.parent.parent)

dir = Path(__file__).resolve()
sys.path.append(str(dir.parent.parent))

path = os.path.dirname(__file__)

def show_title():
    title = "基于CrossFusionSleepNet的睡眠分期辅助系统"
    st.markdown(
      f"<h2 style='text-align: center;'>{title}</h2>", 
      unsafe_allow_html=True
      )


def show_identity():
  col1, col2, col3 = st.columns([5, 5, 5])
  identity = "姓名： | 年龄： |  性别： "
  col2.write(identity)

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
    """支持多轮对话的API调用"""
    host = "127.0.0.1"
    port = "11434"
    llm = Ollama(base_url=f"http://{host}:{port}", model="deepseek-r1:8b", temperature=0)
    
    # 构造完整的对话历史
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # 添加性能监控
    start_time = time.time()
    response = llm.invoke(f"当前对话历史：\n{conversation_history}\nassistant:")
    end_time = time.time()
    
    # 计算响应时间和token生成速度
    elapsed_time = end_time - start_time
    token_count = len(response.split())
    tokens_per_second = token_count / elapsed_time
    
    # 可以选择记录或显示性能数据
    print(f"响应时间: {elapsed_time:.2f}秒, 生成了约{token_count}个token, 速度: {tokens_per_second:.2f} tokens/秒")
    
    return response

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
        
        # 构造系统提示（包含报告数据）
        system_prompt = f"""睡眠分析报告数据：
        {raw_report.to_markdown()}
        
        请根据以上报告数据回答用户问题，回答需：
        1. 结合报告中的具体数据
        2. 使用通俗易懂的语言
        3. 提供专业建议
        当前问题：{user_input}"""
        
        # 生成并添加AI回复
        with st.spinner("正在分析..."):
            response = deepseek_api(st_session.chat_history + [{"role": "user", "content": system_prompt}])
        
        st_session.chat_history.append({"role": "assistant", "content": response})
        
        # 刷新显示
        st.rerun()

# def main():
#   show_title()
#   show_identity()
#   sleep_file  = load_file()

# # 初始化对话历史
#   if "chat_history" not in st_session:
#       st_session.chat_history = []

#   if sleep_file is not None:
#       st.divider()
#       st.markdown(f"<h2 style='text-align: center;'>Sleep Identification Reports</h2>", unsafe_allow_html=True)
      
#       sleep_data, sleep_wave, hypnogram = calculate_sleep_stages(sleep_file)
#       raw_report = generate_report(sleep_data)

#       # 显示初始分析建议
#       if not st_session.chat_history:
#           initial_prompt = f"请分析以下睡眠报告：\n{raw_report.to_markdown()}\n给出总体评估和改善建议"
#           initial_response = deepseek_api([{"role": "user", "content": initial_prompt}])
#           st_session.chat_history.append({"role": "assistant", "content": "## 初始分析建议\n" + initial_response})

#       # 显示聊天界面
#       display_chat_messages()
#       handle_user_input(raw_report)

#       # 显示其他可视化内容（保持原有代码）
#       displaying_sleep_wave(sleep_wave)
#       displaying_hypnogram(hypnogram)
#       displaying_sleep_durations(raw_report)
#       sleep_characteristic()
#       sleep_quality_by_age()



# if __name__ == "__main__":
#   main()

def main():
    show_title()
    show_identity()
    sleep_file = load_file()

    # 初始化对话历史
    if "chat_history" not in st_session:
        st_session.chat_history = []

    if sleep_file is not None:
        st.divider()
        st.markdown(f"<h2 style='text-align: center;'>Sleep Identification Reports</h2>", unsafe_allow_html=True)
        
        # 数据分析和可视化部分
        sleep_data, sleep_wave, hypnogram = calculate_sleep_stages(sleep_file)
        raw_report = generate_report(sleep_data)
        
        # 显示所有可视化内容
        displaying_sleep_wave(sleep_wave)
        displaying_hypnogram(hypnogram)
        displaying_sleep_durations(raw_report)
        sleep_characteristic()
        sleep_quality_by_age()

        st.divider()  # 添加分隔线
        
        # 对话建议部分（放在最后）
        st.markdown("## 睡眠分析建议对话")
        
        # 显示初始分析建议
        if not st_session.chat_history:
            initial_prompt = f"请分析以下睡眠报告：\n{raw_report.to_markdown()}\n给出总体评估和改善建议"
            initial_response = deepseek_api([{"role": "user", "content": initial_prompt}])
            st_session.chat_history.append({"role": "assistant", "content": "## 初始分析建议\n" + initial_response})

        # 显示聊天界面
        display_chat_messages()
        handle_user_input(raw_report)

if __name__ == "__main__":
    main()



