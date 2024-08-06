In this project, we adapt BarrierNet to the UAVs. The CNN-LSTM-MLP part is the perception part. We use ResNet as the CNN-part, taking the sequence of depth images as the input. The output of perception part is the position of obstacles. The output of BarrierNet is the three-axis acceleration. Then, control $\boldsymbol{u}$ is obtained through inverse dynamics.

<img src=".\Pic\framework.png" alt="image-20240226163133648" style="zoom:67%;" />

We plan to test our code in a simple environment as follows. The data collection is finished in https://github.com/Jiang-Yufei/E2E_Flight.git. We will merge this two project and test the algorithm in Gazebo in the future.

<img src=".\Pic\environment.png" alt="image-20240328084005315" style="zoom: 67%;" />