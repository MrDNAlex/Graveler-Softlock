# Graveler Softlock
 A experimental C++ and CUDA program with the goal of running the Graveler Softlock program as fast as possible in the ShoddyCast The Science! YouTube Video ![You'll NEVER Escape ‪@Pikasprey‬'s Evil Soft Lock | The SCIENCE of Soft Lock Picking Pokemon](https://www.youtube.com/watch?v=M8C8dHQE2Ro&t=649s).
 
In the Video, Austin makes a Python program running 1 Billion battle iterations taking 8 days to run. He then challenges the community to get better times.
 
This is my attempt at solving the problem

# Simulation Speed (V3)
![image](https://github.com/user-attachments/assets/573d1106-7b5e-4c30-9581-728f7c4066df)

This time we hardcoded Values, we also limit the Number of cores that end up running, we max out the cores the device uses and each core will do a while loop until all 1 billion calculations are finished

## Results
(60*60*24*8)/2.25 = 691 200 / 2.25 = 307 200 --> 307 200x Speedup

# Simulation Speed (V2)
![Speed](https://github.com/user-attachments/assets/591e87a1-a4ec-4076-bd74-55532d8bfecb)

 ## Results
(60*60*24*8)/69 = 691 200 / 69 = 10017 --> 10 000x Speedup

# Code Snippets (V2)
![SimulateBattleGPU 2](https://github.com/user-attachments/assets/c6914bdb-67cb-4056-824d-ae650f9b4a9c)
![SimulateBattleCPU 2](https://github.com/user-attachments/assets/d81f7024-00cd-430c-aaf9-4a0712fcd90c)

# Simulation Speed (V1)
![Speed](https://github.com/user-attachments/assets/0d5fa987-dea0-4094-8145-11f7b1e6f3a8)

# Code Snippets
![SimulateBattleHeader](https://github.com/user-attachments/assets/6a22f52b-c55c-46b5-9ca7-e0fdfab86b9d)
![SimulateBattleGPU](https://github.com/user-attachments/assets/7353535a-148b-48de-aa55-34abcb1ed585)
![SimulateBattleCPU](https://github.com/user-attachments/assets/9f5a9b07-a06f-4a57-ad61-a219cf62da0a)
![MaxParal](https://github.com/user-attachments/assets/fa703e9d-8896-4733-9996-be71476320a7)
![Main](https://github.com/user-attachments/assets/c0f32299-497a-42c7-8a73-8c7f6dfa6d2b)
![GPUActive](https://github.com/user-attachments/assets/bb3f1fe8-1e1b-464a-89f6-6c32897a2d97)

# Video
https://github.com/user-attachments/assets/78e3b89b-f3e3-4f3b-99b0-1a689ac038af
