myqsubMini mpirun -np 52 ./meas_center_r2.py --minId 0 --maxId 3000 &&
sleep 1 &&
myqsubMini mpirun -np 52 ./meas_center_mag.py --minId 0 --maxId 3000 &&
sleep 1 &&
myqsubMini mpirun -np 52 ./meas_constC.py --minId 0 --maxId 3000 --noirev &&
sleep 1 &&
myqsubMini mpirun -np 52 ./meas_constC.py --minId 0 --maxId 3000 --no-noirev
