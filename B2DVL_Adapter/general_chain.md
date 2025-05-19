PERCEPTION

[18] "What are the important objects in the scene?"

[19] "What are the important objects in the scene? List them from most to least important."

[20] "Where on the road is {vehicle_description} located?"

[26] "What are the important vehicles and where are they on road?"

[27] "The important vehicles are ..., List their locations on road."

[35] "Is the ego vehicle at a junction?"

[34] "How many lanes are there in the {name} direction {to_or_as} the ego car?"

[33] "On which lane is the ego vehicle (left most lane of the lanes going in the same direction is indicated with 0)?"

[32] "What lane marking is on the {name} side of the ego car?"

[5] "Is the ego vehicle affected by a traffic light?"

[1] "How many pedestrians are there?"

[37] "What is current time and weather?"

[11] "Is there an obstacle on the current road?"

[7] "What is the current speed limit?"

PREDICTION

[17] "Where is {other_vehicle_location_description} going?"

[16] "What is the moving status of {other_vehicle_location_description}?"

[21] "What is the rough moving speed and moving direction of {vehicle_description}?"

[22] "What is the exact moving speed and moving direction of {vehicle_description}?"

[24] "The important vehicles are ..., What is the rough moving speed and moving direction of them?"

[25] "The important vehicles are ..., What is the exact moving speed and moving direction of them?"

[36] "The ego vehicle wants to {command_description}. Which lanes are important to watch out for?"

[31] "In which direction is the ego car allowed to change lanes?"

[30] "From which side are other vehicles allowed to change lanes into the ego lane?"

[2] "Is the ego vehicle affected by a stop sign?"

[3] "Is the ego vehicle affected by a speed limit sign?"

[4] "List the traffic signs affecting the ego vehicle in the current scenario."

[6] "What is the state of the traffic light?"

[38] "What is current time and weather? What hazards might it bring?"

[40] "Apart from vehicles on the road, visible pedestrians and the weather, what other factors in the current scenario could pose potential hazards?"

PLANNING

[23] "The ego vehicle {command_str}. Is {vehicle_location_description} potentially crossing the path of the ego vehicle?"

[28] "The important vehicles are ..., Identify potential overlap vehicles and give reasons."

[29] "The important vehicles are ..., List potential overlap vehicles."

[39] "What is current time and weather? What should the ego vehicle do according to them?"

[41] "Apart from vehicles on the road, visible pedestrians and the weather, what other factors in the current scenario could pose potential hazards? What strategies should the ego vehicle adopt to address them?"

[10] "Does the ego vehicle need to change lanes or deviate from the lane center due to an upcoming obstruction?"

[12] "Does the ego vehicle need to change lanes or deviate from the lane for reasons other than the upcoming obstruction? Why?"

[13] "Must the ego vehicle change lane or deviate from the lane now? why?"

[8] "Does the ego vehicle need to brake? Why?"

[9] "What should the ego vehicle do based on the {actor_type}?"

[15] "Identify all traffic lights and signs affecting the ego vehicle in current scene. Based on these traffic signs, what actions should the ego vehicle take respectively?"