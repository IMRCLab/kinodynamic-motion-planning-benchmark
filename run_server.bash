# SERVER=quim
# SERVER=130.149.82.54

USER=quimortiz
SERVER=hal-9000.lis.tu-berlin.de

echo user
echo $USER

echo server
echo $SERVER



time rsync -avr  \
  --exclude build_rwdi/ \
  --exclude build_debug/ \
  --exclude build_fastdebug/ \
  --exclude build_fastdebug_clang/ \
  --exclude build_debug_clang/ \
  --exclude build_clang/ \
  --exclude build/ \
  --exclude build/ \
  --exclude paper/ \
  --exclude results_old/ \
  --exclude build_rwdi/ \
  --exclude build_debug_clang/ \
  --exclude data_welf/ \
  --exclude cloud/motions/  \
  --exclude cloud/motionsX/ \
  --exclude cloud/motions.zip \
  --exclude data_alex/ \
  --exclude data_welf_yaml/ \
  --exclude build_clang/ \
  --exclude build_fastdebug/ \
  --exclude build_debug/ \
  --exclude .git/ \
  --exclude .cache/ \
  /home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/ $USER@$SERVER:~/stg/wolfgang/kinosync

#
# TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
#
# cmd="python3 ../scripts/new_benchmark.py -m bench -bc ../bench/compare.yaml | tee  ../bench_logs/server_'$TIMESTAMP'.z  2>&1"
#
cmd="python3 ../scripts/new_benchmark.py -m bench_time -bc ../bench/bench_time.yaml   | tee  ../bench_logs/server_'$TIMESTAMP'.z  2>&1"
#
#
#
#
# #
# #
# #
# #
#
#

ssh  $USER@$SERVER  << EOF
  cd ~/stg/wolfgang/kinosync/
  source ~/kino/bin/activate.fish
  mkdir bench_logs
  mkdir build
  cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/openrobots/
  make -j40 
  $cmd
EOF

#
time rsync -avr  \
  $USER@$SERVER:~/stg/wolfgang/kinosync/results_new \
  $USER@$SERVER:~/stg/wolfgang/kinosync/results_new_timeopt \
  $USER@$SERVER:~/stg/wolfgang/kinosync/results_new_search \
  $USER@$SERVER:~/stg/wolfgang/kinosync/results_new_components \
  $USER@$SERVER:~/stg/wolfgang/kinosync/bench_logs \
  /home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark
#
