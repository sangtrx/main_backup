#Find JSON that gives errors
JSON_LOC="/home/Desktop/annotation.json"

#Open JSON
val_json = open(JSON_LOC, "r")
json_object = json.load(val_json)
val_json.close()

for i, instance in enumerate(json_object["annotations"]):
    if len(instance["segmentation"][0]) == 4:
        print("instance number", i, "raises arror:", instance["segmentation"][0])