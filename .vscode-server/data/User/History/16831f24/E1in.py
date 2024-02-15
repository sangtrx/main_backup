import json
#Find JSON that gives errors
JSON_LOC="/home/tqsang/V100/tqsang/crop_obj/front_2_class_feather/annotations/instances_train2017.json"

#Open JSON
val_json = open(JSON_LOC, "r")
json_object = json.load(val_json)
val_json.close()

for i, instance in enumerate(json_object["annotations"]):
    if len(instance["segmentation"][0]) == 4:
        print("instance number", i, "raises arror:", instance["segmentation"][0])

#Alter object generating the error with something random not causing the error
# json_object["annotations"][652]["segmentation"] = [[0,0,0,2,3]]

# #Write back altered JSON
# val_json = open(JSON_LOC, "w")
# json.dump(json_object, val_json)
# val_json.close()