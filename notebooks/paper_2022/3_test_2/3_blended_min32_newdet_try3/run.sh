myqsubMini mpirun -np 52 ./meas_detect_r2_2.py --minId 0 --maxId 10000 &&
sleep 1 &&
myqsubMini mpirun -np 52 ./meas_detect_mag_2.py --minId 0 --maxId 10000 &&
sleep 1
# myqsubMini mpirun -np 52 ./meas_detect_r2.py --minId 0 --maxId 10000 &&
# sleep 1 &&
# myqsubMini mpirun -np 52 ./meas_detect_mag.py --minId 0 --maxId 10000
