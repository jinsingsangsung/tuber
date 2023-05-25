import json

print("..........working in val json file.............")
val_json = json.load(open('/home/nsml/assets/ava_{}_v22.json'.format("val"))) 
frame_bbox = val_json["video_frame_bbox"]
frame_keys_list = val_json["frame_keys_list"]
num_bboxes = 0
num_actions = 0
for key in frame_bbox:
    num_bboxes += len(frame_bbox[key]["bboxes"])
    for i in range(len(frame_bbox[key]["bboxes"])):
        num_actions += len(frame_bbox[key]["acts"][i])

print("num_boxes in json file: ", num_bboxes)
print("num_actions in json file: ", num_actions)


val_csv = open('/home/nsml/assets/ava_{}_v2.2.csv'.format("val")).readlines()
num_bboxes_csv = 0
frame_bbox_csv = {
    "video_frame_bbox": {},
    "frame_keys_list": [],
}

acts = []
for i, line in enumerate(val_csv):
    
    img_key = line.split(",")[:2]
    bbox = line.split(",")[2:6]
    bbox = [float(p) for p in bbox]
    act = int(line.split(",")[-2])-1
    actor_id = line.split(",")[-1]
    img_key_ = ",".join(img_key)    
    if i > 1:
        prev_line = val_csv[i-1]
        prev_img_key = prev_line.split(",")[:2]
        prev_bbox = prev_line.split(",")[2:6]
        prev_bbox = [float(p_) for p_ in prev_bbox]
        prev_act = int(prev_line.split(",")[-2])-1
        prev_actor_id = prev_line.split(",")[-1]
        prev_img_key_ = ",".join(prev_img_key)      

        if img_key_ not in frame_bbox_csv["video_frame_bbox"]:
            frame_bbox_csv["video_frame_bbox"][img_key_] = {
                "bboxes":[],
                "acts":[]
            }
            frame_bbox_csv["frame_keys_list"].append(img_key_)

        if not bbox in frame_bbox_csv["video_frame_bbox"][img_key_]["bboxes"] or prev_actor_id != actor_id:
            num_bboxes_csv += 1
            frame_bbox_csv["video_frame_bbox"][img_key_]["bboxes"].append(bbox)
            frame_bbox_csv["video_frame_bbox"][prev_img_key_]["acts"].append(acts)
            acts = [act]
        else:
            # frame_bbox_csv["video_frame_bbox"][img_key_]["acts"].append(act)
            acts.append(act)
    else:
        frame_bbox_csv["video_frame_bbox"][img_key_] = {
            "bboxes":[bbox],
            "acts":[]
        }
        acts.append(act)
        num_bboxes_csv += 1

frame_bbox_csv["video_frame_bbox"][img_key_]["acts"].append(acts)

print("num_boxes in csv file: ", num_bboxes_csv)
print("num_actions in csv file: ", i+1)

num_bboxes_json_only = 0
omitted_frames = []
for key in frame_bbox_csv["video_frame_bbox"]:
    try:
        for box in frame_bbox[key]["bboxes"]:
            if not box in frame_bbox_csv["video_frame_bbox"][key]["bboxes"]:
                num_bboxes_json_only += 1
    except:
        omitted_frames.append(key)

print("number of bboxes that json only has: ", num_bboxes_json_only)
print("number of omitted frames:", len(omitted_frames))


# is omitted frames are made on purpose?

excluded_stamps_val = open('/home/nsml/assets/ava_{}_excluded_timestamps_v2.1.csv'.format("val")).readlines()
excluded_stamps = []
for line in excluded_stamps_val:
    excluded_stamps.append(line[:-1])

valid_omit = 0
for frame in omitted_frames:
    if frame in excluded_stamps:
        valid_omit += 1

print("number of valid omits: ", valid_omit)
# print(omitted_frames)

save_path = "../assets/ava_val_v22_updated.json"

with open(save_path, "w") as f:
    json.dump(frame_bbox_csv, f)
print("renewed val json file has been saved")


print("..........working in train json file.............")
train_json = json.load(open('/home/nsml/assets/ava_{}_v22.json'.format("train"))) 
frame_bbox = train_json["video_frame_bbox"]
frame_keys_list = train_json["frame_keys_list"]
num_bboxes = 0
num_actions = 0
for key in frame_bbox:
    num_bboxes += len(frame_bbox[key]["bboxes"])
    for i in range(len(frame_bbox[key]["bboxes"])):
        num_actions += len(frame_bbox[key]["acts"][i])

print("num_boxes in json file: ", num_bboxes)
print("num_actions in json file: ", num_actions)


train_csv = open('/home/nsml/assets/ava_{}_v2.2.csv'.format("train")).readlines()
num_bboxes_csv = 0
frame_bbox_csv = {
    "video_frame_bbox": {},
    "frame_keys_list": [],
}

acts = []
for i, line in enumerate(train_csv):
    
    img_key = line.split(",")[:2]
    bbox = line.split(",")[2:6]
    bbox = [float(p) for p in bbox]
    act = int(line.split(",")[-2])-1
    actor_id = line.split(",")[-1]
    img_key_ = ",".join(img_key)    
    if i > 1:
        prev_line = train_csv[i-1]
        prev_img_key = prev_line.split(",")[:2]
        prev_bbox = prev_line.split(",")[2:6]
        prev_bbox = [float(p_) for p_ in prev_bbox]
        prev_act = int(prev_line.split(",")[-2])-1
        prev_actor_id = prev_line.split(",")[-1]
        prev_img_key_ = ",".join(prev_img_key)      

        if img_key_ not in frame_bbox_csv["video_frame_bbox"]:
            frame_bbox_csv["video_frame_bbox"][img_key_] = {
                "bboxes":[],
                "acts":[]
            }
            frame_bbox_csv["frame_keys_list"].append(img_key_)

        if not bbox in frame_bbox_csv["video_frame_bbox"][img_key_]["bboxes"] or prev_actor_id != actor_id:
            num_bboxes_csv += 1
            frame_bbox_csv["video_frame_bbox"][img_key_]["bboxes"].append(bbox)
            frame_bbox_csv["video_frame_bbox"][prev_img_key_]["acts"].append(acts)
            acts = [act]
        else:
            # frame_bbox_csv["video_frame_bbox"][img_key_]["acts"].append(act)
            acts.append(act)
    else:
        frame_bbox_csv["video_frame_bbox"][img_key_] = {
            "bboxes":[bbox],
            "acts":[]
        }
        acts.append(act)
        num_bboxes_csv += 1

frame_bbox_csv["video_frame_bbox"][img_key_]["acts"].append(acts)

print("num_boxes in csv file: ", num_bboxes_csv)
print("num_actions in csv file: ", i+1)

num_bboxes_json_only = 0
omitted_frames = []
for key in frame_bbox_csv["video_frame_bbox"]:
    try:
        for box in frame_bbox[key]["bboxes"]:
            if not box in frame_bbox_csv["video_frame_bbox"][key]["bboxes"]:
                num_bboxes_json_only += 1
    except:
        omitted_frames.append(key)

print("number of bboxes that json only has: ", num_bboxes_json_only)
print("number of omitted frames:", len(omitted_frames))


save_path = "../assets/ava_train_v22_updated.json"

with open(save_path, "w") as f:
    json.dump(frame_bbox_csv, f)
print("renewed train json file has been saved")