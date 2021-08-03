import numpy as np
import pandas as pd
import os



def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)



bold_path = "/gpu-data/filby/BoLD/BOLD_public"

categorical_emotions = ["Peace", "Affection", "Esteem", "Anticipation", "Engagement", "Confidence", "Happiness",
                       "Pleasure", "Excitement", "Surprise", "Sympathy", "Doubt/Confusion", "Disconnect",
                       "Fatigue", "Embarrassment", "Yearning", "Disapproval", "Aversion", "Annoyance", "Anger",
                       "Sensitivity", "Sadness", "Disquietment", "Fear", "Pain", "Suffering"]

continuous_emotions = ["Valence", "Arousal", "Dominance"]

attributes = ["Gender", "Age", "Ethnicity"]

header = ["video", "person_id", "min_frame", "max_frame"] + categorical_emotions + continuous_emotions + attributes + ["annotation_confidence"]

df = pd.read_csv(os.path.join(bold_path, "annotations/train.csv"), names=header)

df["joints_path"] = df["video"].apply(rreplace,args=[".mp4",".npy",1])
df["joints25_path"] = df["joints_path"].str.replace("videos","joints_25")


j = []
for i in range(0,df.shape[0]):
	sample = df.iloc[i]
	# joints_path = os.path.join(bold_path, "joints", sample["joints_path"]) # this is to find min max of joints 18
	joints_path = os.path.join(bold_path, "joints_25", sample["joints_path"]) # this is to find min max of joints 18

	if not os.path.exists(joints_path):
	    continue

	joints = np.load(joints_path)
	j.append(joints)

j = np.vstack(j)


from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()
m.fit(j)
print(m.data_min_)
print(m.data_max_)
np.save("data_min",m.data_min_)
np.save("data_max",m.data_max_)