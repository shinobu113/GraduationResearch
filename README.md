# BasicResearch Using HandTracking
This project is a basic research using Google's ML framework called Mediapipe.

# Hand Landmarks
![hand_landmarks](https://user-images.githubusercontent.com/40625062/138540170-baf3951f-6563-4c58-822a-b7780072e8d5.png)

# Environment
- Windows10
- Anaconda 4.10.1
- Web Camera

# Requirement
- matplotlib==3.5.1
- mediapipe==0.8.9.1
- numpy==1.21.6
- opencv-contrib-python==4.5.5.64
- pandas==1.3.5
- Pillow==9.1.0



# Usage
## Pyenv Setup
```
(reference：https://blog.beachside.dev/entry/2020/08/12/190000)

$ makir envs
$ pyenv -m venv envs
$ ./envs/Sctripts/activate
```
## Installs
```
--proxy=http://user:pass@fooooo.proxy.local:8080

(envs) $ pip install --upgrade pip --proxy=~~~
(envs) $ pip install requests --proxy=~~~
(envs) $ pip install mediapipe, loguru --proxy=~~~

```
# VScode Settings
## Settings.json
```
"python.pythonPath": ".\\envs\\Scripts\\python.exe"
"terminal.integrated.profiles.windows":{
        "Command Prompt": {
            "path": [
                "${env:windir}\\Sysnative\\cmd.exe",
                "${env:windir}\\System32\\cmd.exe"
            ],
            "args": [
                "/k",
                ".\\envs\\Scripts\\activate",
            ],
            "icon": "terminal-cmd"
        },
    },
"python.autoComplete.extraPaths": [
        ".\\envs\\Lib\\site-packages"
    ],
"python.analysis.extraPaths": [
        ".\\envs\\Lib\\site-packages",
    ],

```