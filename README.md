# Identify_distracted_driving
Final project of Machine Learning course at CoderSchool

_By Jenny Hoang - 01/2022_

**WHY THIS TOPIC?**
From some pain point

"You’re driving. You hear that familiar noise notifying you of a new text message, so you grab your phone to see what it says. The radio goes to commercial, so you take your hand off the wheel to switch stations. There’s an event outside, so you glance over to see what’s going on." 
==> These are all examples of distracted driving. Fatal distracted driving crashes happen every day.

==> What if there is something that can help to remind drivers when they lose focus on driving real-time like a companion? How about an AI tool?

**THE IDEA**
Solution - My proposal

In this project, I have developed a tool to detect drivers' actions and give alerts if there is anything that takes drivers' attention away from the task of safe driving including talking or texting on phone, eating and drinking, talking to people in the vehicle, fiddling with the stereo, entertainment or navigation system,...
The alert is developed in 2 options: robot sounds or human voices

**THE DATASET:**
From Kaggle and self-collecting

For the phase 1: I used the dataset from Kaggle which includes drivers' images doing something in the car. There are 10 classes: 
  - c0: safe driving
  - c1: texting - right
  - c2: talking on the phone - right
  - c3: texting - left
  - c4: talking on the phone - left
  - c5: operating the radio
  - c6: drinking
  - c7: reaching behind
  - c8: hair and makeup
  - c9: talking to passengers
  After EDA, I decided combine the data of c1 & c3 into "texting" class, and c2 & c4 into "talking" class which means there are only 8 classes. 

For the phase 2:
After having accuracy at 94,18% using MobileNetv2 as pre-trained model, the data of sefl-collecting was used for training and real time testing.

**THE RESULT & DEMO:**
Find the presentation as [LINK](https://www.facebook.com/coderschoolvn/videos/297330995660800/)

**HIGHLIGHTS:**
- The accuracy for real-time detection is ~ 91,48%
- The AI companion can detect different case and give remider in human voice for each case
- When the driver keep doing the action that the AI companion has already given alert, the system would have another sentence to express the reminder to stop the driver's action the back to safe driving

![image](https://user-images.githubusercontent.com/90630291/153575395-7618b33f-486a-4c8c-af02-4b8c132a8f1c.png)
