import os




for line in open("/gpu-data/filby/BoLD/BOLD_public/annotations/train.csv").readlines():
    video_path = os.path.join("/gpu-data/filby/BoLD/BOLD_public/videos", line.split(",")[0])
    print(video_path)
    DIR_OPENPOSE = video_path + "_openface"
    if not os.path.exists(DIR_OPENPOSE):
	    os.mkdir(DIR_OPENPOSE)

    command = "./FaceLandmarkVidMulti -f {video} -out_dir {DIR_OPENPOSE} -simsize 300"
    c = command.format(video=video_path, DIR_OPENPOSE=DIR_OPENPOSE)
    os.system(c)
    # raise
#         print(c)
