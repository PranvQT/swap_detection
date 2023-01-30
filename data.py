from copyreg import add_extension
import pymongo 
from pymongo import MongoClient
import json
import os
import shutil



with open("Settings/config.json","r") as f:
    param_dict = json.load(f)

dir = "../"
directory = param_dict["db_name"]
path = os.path.join(dir,directory)

try: 
    os.mkdir(path) 
except OSError as error: 
    print(directory, "directory already exists, copying json data in", directory) 

conn = MongoClient()
db = conn.quidich_db_v3
db_name = param_dict["db_name"]

dynamicDb_name = db_name + str("_normal_data")
dynamicLivelock_dbName = db_name + str("_livelock_data")
dynamicScoreData_dbName = db_name + str("_score_data")
dynamicHistoryData_dbName = db_name + str("_history_data")
                        
# collection = db[dynamicDb_name]
# collection_livelock = db[dynamicLivelock_dbName]
# score_file_collection = db[dynamicScoreData_dbName]
# score_history_data = db[dynamicHistoryData_dbName]

def save_db(path,collection_name):
    # cursor = collection.find()
    # list_cur = list(cursor)
    # json_data = json.dumps(list_cur, indent = 2)
    os.chdir(path)
    # js = collection_name + ".json"
    os.system("mongoexport --db=quidich_db_v3 --collection=%s --out=%s.json"%(collection_name,collection_name))
    print("copying json data")
    print(os.path)
    os.chdir("../qt-deployment/")
    
    # with open(f'{path}/{collection_name}.json', 'w') as file:
	#     file.write(json_data)
        
save_db(path,dynamicDb_name)
save_db(path,dynamicLivelock_dbName)
save_db(path,dynamicScoreData_dbName)
save_db(path,dynamicHistoryData_dbName) 

config_path = "Settings/config.json"
frame_path = "Settings/frame.jpg"
dst_path = "Settings/dst.jpg"
homo_path = "Settings/cam_params.json"
destination_path = path + "/"
try: 
    shutil.copyfile(config_path,destination_path+"config.json")
    shutil.copyfile(frame_path,destination_path+"frame.jpg")
    shutil.copyfile(dst_path,destination_path+"dst.jpg")
    shutil.copyfile(homo_path,destination_path+"cam_params.json")
    print("config file and frame copied in", directory)
except Exception as e:
    print("Either Config or Scoring file not presentin Settings folder")

if os.path.exists(param_dict["history_data_path"]):
    print("history ball by ball data file is present, copying that")
    try:
        shutil.copyfile(param_dict["history_data_path"],destination_path+"history_ballbyball_data.json")
    except Exception as e:
        print("error copying history file")
else:
    print("history ball by ball data file is not present")

if(param_dict["score_file_mode"] == "wt"):
    scorefile_path = param_dict["score_file_path"]
    try:
        shutil.copyfile(scorefile_path,destination_path+"scorefile.txt")
        print("wt score file copied in", directory)
    except Exception as e:
        print("wt scoring file not present in score_file folder")
elif(param_dict["score_file_mode"]== "ae"):
    scorefile_path = param_dict["score_file_path"]
    aefile_1 = scorefile_path + "/AELInteractiveHawkeye_1.TXT"
    aefile_2 = scorefile_path + "/AELInteractiveHawkeye_2.TXT"
    try:
        shutil.copyfile(aefile_1,destination_path+"AELInteractiveHawkeye_1.TXT")
        shutil.copyfile(aefile_2,destination_path+"AELInteractiveHawkeye_2.TXT")
        print("ae scoring file not present in score_file folder")
    except Exception as e:
        print(e)

try: 
    shutil.make_archive("../"+directory, 'zip', path)
    print("zip file created")
except Exception as e:
    print(e)














