import roboflow


rf = roboflow.Roboflow(api_key="gL#######")

project = rf.workspace("team-roboflow").project("coco-128")
dataset = project.version(2).download("coco")