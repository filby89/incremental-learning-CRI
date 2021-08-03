import os
import cv2



BASE_PATH = "/gpu-data/filby/BoLD"
videos_not_existing = 0
empties = 0
for line in open(os.path.join(BASE_PATH,"BOLD_public/annotations/train.csv")).readlines():
	video_path = os.path.join(BASE_PATH,"BOLD_public/videos", line.split(",")[0])
	DIR_OPENPOSE = video_path + "_openpose"

	if not os.path.exists(video_path):
		videos_not_existing += 1
		continue

	cap = cv2.VideoCapture(video_path)
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	# print( length )

	command = "/openpose/build/examples/openpose/openpose.bin --net_resolution -1x320 --render_pose 0 --video {video} --keypoint_scale 0  --model_pose 'BODY_25' --model_folder /openpose/models/ --face --hand --write_json {DIR_OPENPOSE}/json/ --scale_number 4 --scale_gap 0.25  --display 0 --hand_scale_number 6 --hand_scale_range 0.4"
	c = command.format(video=video_path, DIR_OPENPOSE=DIR_OPENPOSE)

	if not os.path.exists(DIR_OPENPOSE):
		os.mkdir(DIR_OPENPOSE)
	else:
		# print(video_path)

		if len(os.listdir(DIR_OPENPOSE)) == 0:
			empties += 1
			# os.system(c)
			# print(video_path)
			continue

		num_openpose_json = len(os.listdir(DIR_OPENPOSE+"/json"))

		if num_openpose_json == 0:
			empties += 1
			# os.system(c)
			# print(video_path)
			continue

		if (length != num_openpose_json):
			# print(length, num_openpose_json)
			empties += 1
			# print(video_path)
			# print(c)
			continue

		continue
	raise
	# print(c)
	# os.system(c)

print("Not existing", videos_not_existing)
print("Empty", empties)