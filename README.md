
# Sleep-Monitoring-Assistance-System

Created by Cao Yang. 

In view of the current situation of the lack of clinical application of sleep analysis systems, this paper develops a modular sleep staging analysis system. Based on the implemented multi-modal PSG (Polysomnography) sleep staging algorithm, a complete sleep staging analysis system is designed and implemented. This system adopts a modular architecture, which has good extensibility and compatibility. It can provide efficient sleep quality assessment tools for clinicians and also meet the needs of long-term sleep monitoring of household users.  


## Screenshots
![Sleep-Monitoring-Assistance-System website](https://github.com/sa404-star/Sleep-Monitoring-Assistance-System/blob/master/images/app_page-0000.png)
![Sleep-Monitoring-Assistance-System website](https://github.com/sa404-star/Sleep-Monitoring-Assistance-System/blob/master/images/app_page-0001.jpg)
![Sleep-Monitoring-Assistance-System website](https://github.com/sa404-star/Sleep-Monitoring-Assistance-System/blob/master/images/app_page-0002.jpg)
![Sleep-Monitoring-Assistance-System website](https://github.com/sa404-star/Sleep-Monitoring-Assistance-System/blob/master/images/app_page-0003.jpg)
![Sleep-Monitoring-Assistance-System website](https://github.com/sa404-star/Sleep-Monitoring-Assistance-System/blob/master/images/app_page-0004.jpg)
![Sleep-Monitoring-Assistance-System website](https://github.com/sa404-star/Sleep-Monitoring-Assistance-System/blob/master/images/app_page-0005.jpg)


# Main function

- User registration and login
- Presentation of sleep staging results
- Data processing and model integration module
- Sleep phase proportion analysis module
- Sleep analysis and advice module


    

## Run Locally

Clone the project

```bash
  git clone https://github.com/sa404-star/Sleep-Monitoring-Assistance-System.git
```

Go to the project directory

```bash
  cd Sleep-Monitoring-Assistance-System
```

Install dependencies

```bash
  pip install -r requirement.txt
```

Start the project

```bash
  streamlit run app.py
```

## How to use

1. Load your EEG data (in a supported format)
2. The app will automatically process the data and generate the sleep report
3. Explore the report to gain insights into your sleep patterns
4. Explore your sleep, optimize your rest, and uncover the mysteries of your mind!




# reference code
Code reference in[sleep-stages-identification](https://github.com/AjiBegawan/sleep-stages-identification). 
