myqsubMini mpirun -np 52 ./meas_detect_r2.py --minId 0 --maxId 1024 &&
sleep 1 &&
myqsubMini mpirun -np 52 ./meas_detect_mag.py --minId 0 --maxId 1024 &&
sleep 1
