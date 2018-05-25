#   Sensor-Assisted Neural ABR Video Streaming on UAV

The reference implementation for the project

_An Sensor-Assisted Neural ABR Video Streaming on UAV_
was executed during December 2017 to May 2018</br>
by Xuedou Xiao,Taobin Chen,Wei Wang, Tao Jiang ,Qian Zhang

The code is provided only for replication purposes, futher development is not plannde.
If you have questions, pleasr contact [Weisgroup](http://ei.hust.edu.cn/lab/SINC-lab/chineseversion/people/weiwang.html)
or just create an issue here.

## Dependencies

- Python packages :numpy, pylab, math

- [pytorch 0.4.0](https://pytorch.org/)

## How to use

- Install all the dependencies(see the list above)

- If you want to know the original throughput, speed, acceleration and distance information between the UAV and the receiver, which is 

measured by the sensor of UAV and received by the wireshark, you can see the “raw data" folder. And if you are curious about the traiing 

data we exact from the raw data, you can go straight to the "training data" folder.

- For training, you can run : 
  
  >baseline.py for the traditional algorithm ;
  
  >trainCNN_without_sensor.py for the DRL algorithm with the throughput  data implemented by the CNN network;
  
  >trainCNN_with_sensor.py for the DRL algorithm with the throughput and sensor data implemented by the CNN netork;
  
  >trainLSTM_with_sensor.py for the DRL algorithm with the throughput and sensor data implemented by the RNN netork.
 
- For the model we trained, see them at the "model" folder.

## License

MIT
