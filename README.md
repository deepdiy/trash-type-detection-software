# Trash Type Detecion Software


Classify trash by computer vision. A ssd-mobilenet based deep learning model and a kivy based GUI is included in this repository. Currently, the software can detect bottles in video frame.

![demo](https://i.imgur.com/wFttTnb.png)

![demo](https://i.imgur.com/QD2C5TC.jpg)

![demo2](https://i.imgur.com/HYl1Z2c.jpg)

# Installation
```python
pip install -r requirements.txt
```

# Run App
```python
python app/main.py
```

# Capture image without detection

1. Run App
2. Click `Capture` button in window

# Training model
[Follow this instruction](https://github.com/deepdiy/trash-type-detection-software/tree/master/tf_ssd_mobilenet)

# Use new model
put `frozen_inference_graph.pb` in `/app/model`
