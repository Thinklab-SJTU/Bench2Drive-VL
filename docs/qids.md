All supported questions. (Ones with \* is originally supported by [DriveLM-CARLA](https://github.com/OpenDriveLab/DriveLM/tree/DriveLM-CARLA) open-source annotation script)

1. \* How many pedestrians are there?  
2. \* Is the ego vehicle affected by a stop sign?  
3. Is the ego vehicle affected by a speed limit sign?  
4. List the traffic signs affecting the ego vehicle in the current scenario.  
5. \* Is the ego vehicle affected by a traffic light?  
6. \* What is the state of the traffic light?  
7. What is the current speed limit?  
8. \* Does the ego vehicle need to brake? Why?  
9. \* What should the ego vehicle do based on the `\{actor_type\}`?  
10. \* Does the ego vehicle need to change lanes or deviate from the lane center due to an upcoming obstruction?  
11. \* Is there an obstacle on the current road?  
12. Does the ego vehicle need to change lanes or deviate from the lane for reasons other than the upcoming obstruction? Why?  
13. Must the ego vehicle change lane or deviate from the lane now? Why?  
14. The list of traffic lights and signs affecting the ego vehicle in current scene is: `\{sign_list_str\}`. Based on these traffic signs, what actions should the ego vehicle take respectively?  
15. Identify all traffic lights and signs affecting the ego vehicle in current scene. Based on these traffic signs, what actions should the ego vehicle take respectively?  
16. \* What is the moving status of `\{other_vehicle_location_description\}`?  
17. \* Where is `\{other_vehicle_location_description\}` going?  
18. \* What are the important objects in the scene?  
19. What are the important objects in the scene? List them from most to least important.  
20. \* Where on the road is `\{vehicle_description\}` located?  
21. What is the rough moving speed and moving direction of `\{vehicle_description\}`?  
22. What is the exact moving speed and moving direction of `\{vehicle_description\}`?  
23. The ego vehicle `\{command_str\}`. Is `\{vehicle_location_description\}` potentially crossing the path of the ego vehicle? If so, why?  
24. The important vehicles are ..., What is the rough moving speed and moving direction of them?  
25. The important vehicles are ..., What is the exact moving speed and moving direction of them?  
26. \* What are the important vehicles and where are they on road?  
27. The important vehicles are ..., List their locations on road.  
28. The important vehicles are ..., Identify potential overlap vehicles and give reasons.  
29. The important vehicles are ..., List potential overlap vehicles.  
30. \* From which side are other vehicles allowed to change lanes into the ego lane?  
31. \* In which direction is the ego car allowed to change lanes?  
32. \* What lane marking is on the `\{name\}` side of the ego car?  
33. \* On which lane is the ego vehicle (left most lane of the lanes going in the same direction is indicated with 0)?  
34. \* How many lanes are there in the `\{name\}` direction `\{to_or_as\}` the ego car?  
35. \* Is the ego vehicle at a junction?  
36. \* The ego vehicle wants to `\{command_description\}`. Which lanes are important to watch out for?  
37. What is current time and weather?  
38. What is current time and weather? What hazards might it bring?  
39. What is current time and weather? What should the ego vehicle do according to them?  
40. Apart from vehicles on the road, visible pedestrians and the weather, what other factors in the current scenario could pose potential hazards?  
41. Apart from vehicles on the road, visible pedestrians and the weather, what other factors in the current scenario could pose potential hazards? What strategies should the ego vehicle adopt to address them?  
42. Predict the ego vehicle's future waypoint...  
43. What is the correct action for the ego vehicle to take now?  
44. Describe the current lane's direction.  
45. （index 45 is left blank）  
46. The important vehicles are ..., List potential overlap vehicles and the actions that could lead to a collision.  
47. The important vehicles are ..., List potential overlap vehicles, overlap reasons and the actions that could lead to a collision.  
48. The ego vehicle `\{command_str\}`. Is `\{vehicle_description_with_location\}` potentially crossing the path of the ego vehicle? If so, why? And what action can lead to a collision?  
49. The ego vehicle `\{command_str\}`. Is `\{vehicle_description_with_location\}` potentially crossing the path of the ego vehicle? If so, what action can lead to a collision?  
50. Provide the appropriate behavior for the ego vehicle, `FOLLOW_LANE`, `CHANGE_LANE_LEFT`, `CHANGE_LANE_RIGHT`, `GO_STRAIGHT`, `TURN_LEFT`, or `TURN_RIGHT` and the Speed key, which can be `KEEP`, `ACCELERATE`, `DECELERATE`, or `STOP`.